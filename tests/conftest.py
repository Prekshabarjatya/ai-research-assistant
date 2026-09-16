import pytest

import app.main as main
from app.config import settings
from app.vectorstore import VectorStore


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    """Each test's TestClient triggers the startup event, which re-indexes
    sample_data/ into `app.main.store`. Without a reset that store is one
    shared, ever-growing global across the whole test session — reassigning
    it here before every test keeps tests isolated from each other.

    Also forces `groq_api_key` empty regardless of a local `.env`: tests
    that exercise the "no LLM configured" path shouldn't pass or fail
    depending on whether the developer running them happens to have a real
    key on disk. Tests for the configured path monkeypatch
    `main.is_configured` / `main.synthesize` explicitly instead of relying
    on a real key."""
    monkeypatch.setattr(settings, "groq_api_key", "")
    main.store = VectorStore()
    yield
    main.store = VectorStore()
