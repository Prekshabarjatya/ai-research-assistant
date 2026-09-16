"""Turns a source file into normalized text ready for chunking.

Handles the three formats the service accepts: plain text, Markdown (kept
as-is — its structure is exactly the paragraph/heading boundaries the
chunker already prefers to split on), and PDF (extracted page by page and
rejoined, since a PDF's layout carries no paragraph markup of its own).
"""

import io
import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}


def extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return raw.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf(raw)
    raise ValueError(
        f"Unsupported file type {suffix!r}. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _extract_pdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def load_sample_corpus(directory: Path) -> list[tuple[str, str]]:
    """Reads every supported file directly under `directory`.

    Returns (document_name, text) pairs, skipping anything that fails to
    parse rather than failing startup over one bad file.
    """
    if not directory.exists():
        return []

    documents: list[tuple[str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = extract_text(path.name, path.read_bytes())
        except Exception:
            logger.exception("Skipping %s: failed to extract text", path.name)
            continue
        if text.strip():
            documents.append((path.name, text))
    return documents
