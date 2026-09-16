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


def test_health_reports_web_search_status():
    with make_client() as client:
        resp = client.get("/health")
    assert "web_search_configured" in resp.json()


def test_ingest_text_with_tags_then_filter_documents_by_tag():
    with make_client() as client:
        client.post(
            "/api/ingest/text",
            json={"title": "roadmap.md", "text": "Q3 priorities and milestones.", "tags": ["planning", "internal"]},
        )
        client.post(
            "/api/ingest/text",
            json={"title": "recipe.md", "text": "How to make sourdough bread.", "tags": ["cooking"]},
        )

        all_docs = client.get("/api/documents").json()
        planning_only = client.get("/api/documents", params={"tag": "planning"}).json()

    roadmap = next(d for d in all_docs["documents"] if d["document"] == "roadmap.md")
    assert set(roadmap["tags"]) == {"planning", "internal"}
    assert "planning" in all_docs["tags"]
    assert "cooking" in all_docs["tags"]

    planning_names = {d["document"] for d in planning_only["documents"]}
    assert planning_names == {"roadmap.md"}


def test_ingest_file_accepts_comma_separated_tags():
    with make_client() as client:
        resp = client.post(
            "/api/ingest/file",
            files={"file": ("notes.txt", b"Some plain text notes.", "text/plain")},
            data={"tags": "team-a, urgent ,"},
        )
        docs = client.get("/api/documents").json()

    assert resp.status_code == 200
    notes = next(d for d in docs["documents"] if d["document"] == "notes.txt")
    assert set(notes["tags"]) == {"team-a", "urgent"}


def test_research_without_groq_key_returns_503():
    with make_client() as client:
        resp = client.post("/api/research", json={"question": "What separators does the chunker try first?"})
    assert resp.status_code == 503
    assert "GROQ_API_KEY" in resp.json()["detail"]


def test_research_runs_graph_and_returns_report(monkeypatch):
    monkeypatch.setattr(main, "is_configured", lambda: True)
    monkeypatch.setattr(main, "web_search_configured", lambda: False)

    def fake_run_research(question, local_sources, use_web):
        return {
            "report": "The chunker tries paragraph breaks first.\n\n---\n**Review:** Verified against sources.",
            "verified": True,
            "review_notes": "",
            "revision_count": 0,
            "local_sources": local_sources,
            "web_sources": [],
            "web_search_error": None,
        }

    monkeypatch.setattr(main, "run_research", fake_run_research)

    with make_client() as client:
        resp = client.post("/api/research", json={"question": "What separators does the chunker try first?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is True
    assert "paragraph breaks" in body["report"]
    assert body["web_search_used"] is False
    assert len(body["sources"]) > 0  # local retrieval ran before the graph


def test_research_pipeline_failure_returns_502(monkeypatch):
    monkeypatch.setattr(main, "is_configured", lambda: True)

    def boom(question, local_sources, use_web):
        raise RuntimeError("groq timeout")

    monkeypatch.setattr(main, "run_research", boom)

    with make_client() as client:
        resp = client.post("/api/research", json={"question": "anything"})

    assert resp.status_code == 502
    assert "groq timeout" in resp.json()["detail"]
