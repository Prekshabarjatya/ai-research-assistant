"""Tests the LangGraph research pipeline with the LLM and web search mocked
out — no real Groq or Tavily calls, so this needs no API keys."""

from unittest.mock import MagicMock

import app.agents.graph as graph_module
from app.agents.graph import build_graph


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _patch_llm(monkeypatch, responses: list[str]):
    """Each call to `_llm().invoke(...)` returns the next response in order."""
    calls = iter(responses)

    def fake_llm():
        client = MagicMock()
        client.invoke.side_effect = lambda _messages: FakeResponse(next(calls))
        return client

    monkeypatch.setattr(graph_module, "_llm", fake_llm)


def _local_sources():
    return [
        {
            "kind": "local",
            "title": "chunking-strategy.md",
            "reference": "chunking-strategy.md, chunk 0",
            "text": "The splitter tries paragraph breaks first, then sentences, then words.",
        }
    ]


def test_verified_on_first_pass_skips_revision(monkeypatch):
    monkeypatch.setattr(graph_module, "web_search_configured", lambda: False)
    _patch_llm(
        monkeypatch,
        [
            "The chunker tries paragraph breaks first.\n\n## Key findings\n- Paragraph breaks come first (source: 1).\n\n## Synthesis\nStructure-aware splitting.",
            "VERIFIED",
        ],
    )

    graph = build_graph()
    result = graph.invoke(
        {
            "question": "What separators does the chunker try first?",
            "use_web": False,
            "local_sources": _local_sources(),
            "web_sources": [],
            "web_search_error": None,
            "draft": "",
            "review_notes": "",
            "verified": False,
            "revision_count": 0,
            "report": "",
        }
    )

    assert result["verified"] is True
    assert result["revision_count"] == 0
    assert "Paragraph breaks come first" in result["report"]
    assert "Verified against sources" in result["report"]


def test_needs_revision_loops_back_to_write_once(monkeypatch):
    monkeypatch.setattr(graph_module, "web_search_configured", lambda: False)
    _patch_llm(
        monkeypatch,
        [
            "First draft with an uncited claim.",  # write (pass 1)
            "NEEDS REVISION\n- The claim isn't cited.",  # review (pass 1)
            "Second draft, now cited (source: 1).",  # write (pass 2, after revision)
            "VERIFIED",  # review (pass 2)
        ],
    )

    graph = build_graph()
    result = graph.invoke(
        {
            "question": "What separators does the chunker try first?",
            "use_web": False,
            "local_sources": _local_sources(),
            "web_sources": [],
            "web_search_error": None,
            "draft": "",
            "review_notes": "",
            "verified": False,
            "revision_count": 0,
            "report": "",
        }
    )

    assert result["revision_count"] == 1
    assert "Second draft" in result["report"]
    assert result["verified"] is True


def test_revision_budget_caps_the_loop(monkeypatch):
    monkeypatch.setattr(graph_module, "web_search_configured", lambda: False)
    # Every review comes back NEEDS REVISION; the graph must still terminate.
    _patch_llm(
        monkeypatch,
        [
            "Draft 1.",
            "NEEDS REVISION\n- issue",
            "Draft 2.",
            "NEEDS REVISION\n- issue again",
        ],
    )
    monkeypatch.setattr(graph_module.settings, "max_research_revisions", 1)

    graph = build_graph()
    result = graph.invoke(
        {
            "question": "q",
            "use_web": False,
            "local_sources": _local_sources(),
            "web_sources": [],
            "web_search_error": None,
            "draft": "",
            "review_notes": "",
            "verified": False,
            "revision_count": 0,
            "report": "",
        }
    )

    assert result["revision_count"] == 1
    assert result["verified"] is False
    assert "Draft 2" in result["report"]
    assert "some claims may be unsupported" in result["report"]


def test_gather_includes_web_sources_when_configured(monkeypatch):
    from app.tools.web_search import WebResult

    monkeypatch.setattr(graph_module, "web_search_configured", lambda: True)
    monkeypatch.setattr(
        graph_module,
        "web_search",
        lambda question, **kw: [WebResult(title="Example", url="https://example.com", content="Example content.")],
    )

    state = {
        "question": "anything",
        "use_web": True,
        "local_sources": [],
        "web_sources": [],
        "web_search_error": None,
        "draft": "",
        "review_notes": "",
        "verified": False,
        "revision_count": 0,
        "report": "",
    }
    result = graph_module.gather(state)

    assert result["web_sources"] == [
        {"kind": "web", "title": "Example", "reference": "https://example.com", "text": "Example content."}
    ]
    assert result["web_search_error"] is None


def test_gather_swallows_web_search_failure(monkeypatch):
    from app.tools.web_search import WebSearchError

    monkeypatch.setattr(graph_module, "web_search_configured", lambda: True)

    def boom(question, **kw):
        raise WebSearchError("bad key")

    monkeypatch.setattr(graph_module, "web_search", boom)

    state = {
        "question": "anything",
        "use_web": True,
        "local_sources": [],
        "web_sources": [],
        "web_search_error": None,
        "draft": "",
        "review_notes": "",
        "verified": False,
        "revision_count": 0,
        "report": "",
    }
    result = graph_module.gather(state)

    assert result["web_sources"] == []
    assert result["web_search_error"] == "bad key"
