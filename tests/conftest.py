import pytest

import app.main as main
from app.vectorstore import VectorStore


@pytest.fixture(autouse=True)
def fresh_store():
    """Each test's TestClient triggers the startup event, which re-indexes
    sample_data/ into `app.main.store`. Without a reset that store is one
    shared, ever-growing global across the whole test session — reassigning
    it here before every test keeps tests isolated from each other."""
    main.store = VectorStore()
    yield
    main.store = VectorStore()
