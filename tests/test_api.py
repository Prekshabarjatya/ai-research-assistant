import app.main as main
from app.vectorstore import SearchHit
from fastapi.testclient import TestClient


def make_client():
    return TestClient(main.app)


def test_health_reports_index_and_groq_status():
    with make_client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["groq_configured"], bool)
    assert body["documents_indexed"] >= 3  # sample_data/ ships three documents
    assert body["chunks_indexed"] > 0


def test_query_with_no_relevant_passage_returns_empty_sources():
    with make_client() as client:
        resp = client.post("/api/query", json={"question": "purple giraffe underwater tango festival"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] is None
    assert body["grounded"] is False
    assert body["sources"] == []
    assert body["note"]


def test_query_without_groq_key_returns_passages_not_a_synthesized_answer():
    with make_client() as client:
        resp = client.post("/api/query", json={"question": "What separators does the chunker try first?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is False
    assert body["answer"] is None
    assert "GROQ_API_KEY" in body["note"]
    assert len(body["sources"]) > 0
    assert body["sources"][0]["score"] > 0


def test_query_with_synthesis_configured_returns_grounded_answer(monkeypatch):
    monkeypatch.setattr(main, "is_configured", lambda: True)
    monkeypatch.setattr(main, "synthesize", lambda question, hits: "The chunker tries paragraph breaks first (source: 1).")

    with make_client() as client:
        resp = client.post("/api/query", json={"question": "What separators does the chunker try first?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["note"] is None
    assert "paragraph breaks" in body["answer"]
    assert len(body["sources"]) > 0


def test_ingest_text_then_retrieve_it():
    with make_client() as client:
        ingest_resp = client.post(
            "/api/ingest/text",
            json={"title": "secret-codes.md", "text": "The secret launch code is zephyrwave42."},
        )
        assert ingest_resp.status_code == 200
        assert ingest_resp.json()["chunks_added"] >= 1

        query_resp = client.post("/api/query", json={"question": "What is the secret launch code?"})

    body = query_resp.json()
    documents = {s["document"] for s in body["sources"]}
    assert "secret-codes.md" in documents


def test_ingest_rejects_unsupported_file_type():
    with make_client() as client:
        resp = client.post(
            "/api/ingest/file",
            files={"file": ("malware.exe", b"not really a document", "application/octet-stream")},
        )
    assert resp.status_code == 415


def test_documents_endpoint_lists_sample_corpus():
    with make_client() as client:
        resp = client.get("/api/documents")
    assert resp.status_code == 200
    names = {d["document"] for d in resp.json()["documents"]}
    assert "architecture.md" in names


def test_shared_password_gate_when_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "app_password", "letmein")
    with make_client() as client:
        unauthenticated = client.get("/api/documents")
        assert unauthenticated.status_code == 401

        authenticated = client.get("/api/documents", auth=("ignored-username", "letmein"))
        assert authenticated.status_code == 200

        health = client.get("/health")
        assert health.status_code == 200  # health check is always open
