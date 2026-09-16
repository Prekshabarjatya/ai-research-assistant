"""In-process vector index over TF-IDF passage vectors.

Retrieval only needs one contract: turn a query into a vector, turn every
stored passage into a vector in the same space, and rank passages by
similarity. Nothing downstream of that contract cares whether the vectors
came from a neural embedding model or a sparse term-frequency one — so this
module is the only place that decision lives.

This deployment uses scikit-learn's TF-IDF vectorizer rather than a dense
sentence-embedding model: it needs no GPU, no model download, and no API
key, so ingestion and retrieval both work out of the box. A corpus small
enough for a demo service doesn't need more than that — see the README's
"Design notes" for what actually pushes a service toward dense embeddings
and an ANN index, and where the swap point is.

TF-IDF also means the whole index has to be refit whenever new text is
added (a term's weight depends on how rare it is across every document, so
one new document can shift every existing vector). Fine at this scale;
a corpus large enough to make that refit slow is also large enough to
justify swapping in an incremental index.
"""

import threading
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.chunking import chunk_text


@dataclass
class Passage:
    document: str
    chunk_index: int
    text: str


@dataclass
class SearchHit:
    document: str
    chunk_index: int
    text: str
    score: float


@dataclass
class VectorStore:
    passages: list[Passage] = field(default_factory=list)
    document_tags: dict[str, list[str]] = field(default_factory=dict)
    _vectorizer: TfidfVectorizer | None = field(default=None, repr=False)
    _matrix: np.ndarray | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_document(self, document: str, text: str, tags: list[str] | None = None) -> int:
        """Chunks `text` and adds it under `document`. Returns chunks added."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        with self._lock:
            start = len(self.passages)
            self.passages.extend(
                Passage(document=document, chunk_index=start + i, text=chunk)
                for i, chunk in enumerate(chunks)
            )
            # Re-ingesting a document under the same name replaces its tags
            # rather than merging — the new ingest is the source of truth.
            self.document_tags[document] = sorted({t.strip() for t in (tags or []) if t.strip()})
            self._refit()
        return len(chunks)

    def _refit(self) -> None:
        corpus = [p.text for p in self.passages]
        # max_df is deliberately left at its default (1.0, no filtering):
        # with a handful of passages, an aggressive max_df can end up
        # excluding every term ("max_df corresponds to < documents than
        # min_df"). stop_words already screens out the overly-common terms
        # max_df would otherwise target.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            norm="l2",
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int) -> list[SearchHit]:
        with self._lock:
            if not self.passages or self._vectorizer is None:
                return []
            query_vec = self._vectorizer.transform([query])
            # Both matrices are L2-normalized, so the dot product is cosine
            # similarity directly — no separate normalization step needed.
            scores = (self._matrix @ query_vec.T).toarray().ravel()
            passages = list(self.passages)

        if not scores.size:
            return []

        top_idx = np.argsort(-scores)[:k]
        return [
            SearchHit(
                document=passages[i].document,
                chunk_index=passages[i].chunk_index,
                text=passages[i].text,
                score=float(scores[i]),
            )
            for i in top_idx
            if scores[i] > 0
        ]

    @property
    def document_names(self) -> list[str]:
        seen: dict[str, int] = {}
        for p in self.passages:
            seen[p.document] = seen.get(p.document, 0) + 1
        return list(seen.keys())

    def document_summary(self, tag: str | None = None) -> list[tuple[str, int, list[str]]]:
        counts: dict[str, int] = {}
        for p in self.passages:
            counts[p.document] = counts.get(p.document, 0) + 1
        summary = [(name, n, self.document_tags.get(name, [])) for name, n in counts.items()]
        if tag:
            summary = [row for row in summary if tag in row[2]]
        return summary

    @property
    def all_tags(self) -> list[str]:
        seen: set[str] = set()
        for tags in self.document_tags.values():
            seen.update(tags)
        return sorted(seen)

    @property
    def chunk_count(self) -> int:
        return len(self.passages)
