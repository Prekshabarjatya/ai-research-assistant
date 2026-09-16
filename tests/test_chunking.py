from app.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_a_single_chunk():
    text = "This is one short paragraph that easily fits in a single chunk."
    chunks = chunk_text(text, chunk_size=900, chunk_overlap=150)
    assert chunks == [text]


def test_long_text_splits_into_multiple_chunks_with_overlap():
    paragraph = "Sentence number {n} in the passage. "
    text = "\n\n".join(paragraph.format(n=n) * 4 for n in range(20))

    chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 for c in chunks)  # splitter may slightly overshoot at a hard break
    # Overlap: the tail of one chunk should share content with the head of the next.
    for first, second in zip(chunks, chunks[1:]):
        tail_words = first.split()[-3:]
        assert any(word in second for word in tail_words)


def test_chunks_are_stripped_and_nonempty():
    text = "\n\n".join(f"Paragraph {n} has its own complete sentence here." for n in range(10))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk == chunk.strip()
        assert chunk != ""
