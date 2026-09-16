from app.vectorstore import VectorStore


def test_empty_store_returns_no_hits():
    store = VectorStore()
    assert store.search("anything", k=4) == []


def test_search_ranks_the_more_relevant_document_first():
    store = VectorStore()
    store.add_document(
        "gardening.md",
        "Tomato plants need at least six hours of direct sunlight a day. "
        "Water tomato seedlings deeply but infrequently to encourage deep roots.",
    )
    store.add_document(
        "astronomy.md",
        "Neutron stars are the collapsed core of a massive supergiant star. "
        "A teaspoon of neutron star material would weigh billions of tons.",
    )

    hits = store.search("how much sunlight do tomato seedlings need", k=2)

    assert len(hits) >= 1
    assert hits[0].document == "gardening.md"
    assert hits[0].score > 0


def test_add_document_returns_chunk_count_and_updates_totals():
    store = VectorStore()
    added = store.add_document("notes.md", "One short paragraph about nothing in particular.")
    assert added >= 1
    assert store.chunk_count == added
    assert store.document_names == ["notes.md"]


def test_ingesting_a_second_document_does_not_lose_the_first():
    store = VectorStore()
    store.add_document("a.md", "Alpha document content about apples.")
    store.add_document("b.md", "Beta document content about bananas.")

    hits = store.search("apples", k=5)
    documents_seen = {h.document for h in hits}
    assert "a.md" in documents_seen
