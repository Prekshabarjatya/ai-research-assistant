# System architecture

This service answers questions about a document corpus using retrieval-augmented
generation (RAG). It has two pipelines that share a vector index.

## Ingestion pipeline (offline, once per document)

1. **Extract** — a source file (Markdown, plain text, or PDF) is read and
   normalized into plain text. PDF pages are extracted individually and
   rejoined with paragraph breaks, since a PDF carries no paragraph markup
   of its own.
2. **Chunk** — the normalized text is split into overlapping passages using
   a recursive character splitter. It tries to break on paragraph
   boundaries first, then sentences, then words, so a passage rarely opens
   or closes mid-sentence. Default chunk size is 900 characters with a
   150-character overlap between consecutive chunks.
3. **Index** — every chunk from every document is vectorized together and
   added to the in-process vector store.

## Query pipeline (online, per request)

1. **Vectorize the question** using the same vectorizer fit on the stored
   passages, so the query lands in the same vector space without a
   translation step.
2. **Retrieve** the passages with the highest cosine similarity to the
   query vector — by default the top 4, filtered to a minimum relevance
   score so an unrelated question doesn't retrieve irrelevant filler.
3. **Synthesize** an answer by prompting a language model with only the
   retrieved passages, numbered, and instructing it to answer solely from
   them and to cite which passage number it drew from. If no language
   model backend is configured, the retrieved passages are returned
   directly instead of a synthesized answer.

## Why TF-IDF instead of a neural embedding model

The vector index uses scikit-learn's TF-IDF vectorizer rather than a dense
sentence-embedding model. That keeps the service dependency-light: no GPU,
no model download, no embedding API key, and ingestion works the moment the
service starts. The retrieval interface (`app/vectorstore.py`) doesn't
care how a vector was produced, so swapping in a dense embedding model
later is a change to one file, not a redesign.

## Where the language model comes in

Synthesis is the only step that calls an external API — a Groq-hosted
model, used through LangChain. Retrieval and generation are deliberately
decoupled: the retriever has no knowledge of which model reads its output,
so either half can be swapped independently.
