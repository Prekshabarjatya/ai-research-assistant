"""Web search via Tavily, used by the research graph's gather step.

Talks to the official `tavily-python` client directly rather than through
a LangChain wrapper — the graph only needs a plain list of results, so a
thin wrapper here keeps the dependency surface small and the return shape
under our control.
"""

from dataclasses import dataclass

from tavily import TavilyClient

from app.config import settings


class WebSearchError(RuntimeError):
    """Raised when the Tavily API call itself fails (bad key, network, rate limit)."""


@dataclass
class WebResult:
    title: str
    url: str
    content: str


def is_configured() -> bool:
    return bool(settings.tavily_api_key)


def web_search(query: str, max_results: int | None = None) -> list[WebResult]:
    if not is_configured():
        raise WebSearchError("TAVILY_API_KEY is not configured.")

    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        response = client.search(
            query,
            max_results=max_results or settings.web_search_max_results,
            search_depth="basic",
            include_answer=False,
        )
    except Exception as exc:  # bad key, network error, rate limit, etc.
        raise WebSearchError(str(exc)) from exc

    return [
        WebResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
        )
        for r in response.get("results", [])
    ]
