"""Splits normalized document text into overlapping passages.

Uses a recursive character splitter: it tries to break on paragraph
boundaries first, then sentences, then words, only falling back to a hard
character cut if nothing smaller fits. That keeps a chunk from opening or
closing mid-sentence whenever the source text gives it a cleaner place to
break.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
