from unittest.mock import MagicMock

import app.tools.web_search as web_search_module
from app.tools.web_search import WebSearchError, is_configured, web_search


def test_is_configured_reflects_settings(monkeypatch):
    monkeypatch.setattr(web_search_module.settings, "tavily_api_key", "")
    assert is_configured() is False
    monkeypatch.setattr(web_search_module.settings, "tavily_api_key", "tvly-fake")
    assert is_configured() is True


def test_web_search_without_key_raises(monkeypatch):
    monkeypatch.setattr(web_search_module.settings, "tavily_api_key", "")
    try:
        web_search("test query")
        assert False, "expected WebSearchError"
    except WebSearchError as exc:
        assert "TAVILY_API_KEY" in str(exc)


def test_web_search_maps_results(monkeypatch):
    monkeypatch.setattr(web_search_module.settings, "tavily_api_key", "tvly-fake")

    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [
            {"title": "A page", "url": "https://example.com/a", "content": "Some content."},
            {"title": "", "url": "https://example.com/b", "content": "More content."},
        ]
    }
    monkeypatch.setattr(web_search_module, "TavilyClient", lambda api_key: fake_client)

    results = web_search("test query", max_results=2)

    assert len(results) == 2
    assert results[0].title == "A page"
    assert results[0].url == "https://example.com/a"
    assert results[1].title == ""


def test_web_search_wraps_client_exception(monkeypatch):
    monkeypatch.setattr(web_search_module.settings, "tavily_api_key", "tvly-fake")

    fake_client = MagicMock()
    fake_client.search.side_effect = RuntimeError("rate limited")
    monkeypatch.setattr(web_search_module, "TavilyClient", lambda api_key: fake_client)

    try:
        web_search("test query")
        assert False, "expected WebSearchError"
    except WebSearchError as exc:
        assert "rate limited" in str(exc)
