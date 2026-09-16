"""Three-agent research pipeline: gather, write, review.

    gather --> write --> review --(needs revision, budget left)--> write
                            \\--(verified, or budget spent)--> finalize --> END

`gather` gets the web half of the source list (local retrieval already ran
before the graph was invoked — see app/main.py — so this node's only job is
Tavily). `write` drafts a cited markdown report from every source. `review`
re-reads the draft against the same sources looking for claims that
outrun their citation, and either sends it back to `write` with concrete
feedback or lets it through. The loop is capped by
`settings.max_research_revisions` so a stubborn draft can't cycle forever.

Each node calls the language model independently — this is genuinely three
passes over the problem with three different jobs, not one call wearing
three hats.
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.tools.web_search import WebSearchError, is_configured as web_search_configured, web_search


class SourceDict(TypedDict):
    kind: str  # "local" | "web"
    title: str
    reference: str  # "document.md, chunk 3" for local; the URL for web
    text: str


class ResearchState(TypedDict):
    question: str
    use_web: bool
    local_sources: list[SourceDict]
    web_sources: list[SourceDict]
    web_search_error: str | None
    draft: str
    review_notes: str
    verified: bool
    revision_count: int
    report: str


WRITE_SYSTEM_PROMPT = """You are a research writer. Using ONLY the numbered sources given, write a \
structured markdown report that answers the question. Follow this shape exactly:

1. A one-sentence direct answer as the opening line.
2. A "## Key findings" section: 2-5 bullet points, each ending with a citation \
like (source: N).
3. A "## Synthesis" section: one short paragraph connecting the findings.

Cite every factual claim with (source: N). Never state something the sources \
don't support, and never invent a source number. If the sources conflict, say so \
instead of picking one silently."""

REVIEW_SYSTEM_PROMPT = """You are a fact-checking reviewer. You will be given numbered sources and a \
draft report that cites them. Check every factual claim in the draft against the \
source it cites.

Respond in exactly this format:
Line 1: the single word VERIFIED, or the phrase NEEDS REVISION
If NEEDS REVISION, follow with one bullet per problem: name the claim and say \
whether it's uncited, miscited, or unsupported by what that source actually says."""


def _numbered_sources(state: ResearchState) -> list[tuple[int, SourceDict]]:
    combined = [*state["local_sources"], *state["web_sources"]]
    return list(enumerate(combined, start=1))


def _format_sources(numbered: list[tuple[int, SourceDict]]) -> str:
    lines = []
    for num, src in numbered:
        lines.append(f"[{num}] ({src['kind']}) {src['title']} — {src['reference']}\n{src['text']}")
    return "\n\n".join(lines) if lines else "(no sources retrieved)"


def _llm() -> ChatGroq:
    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.2)


def gather(state: ResearchState) -> dict:
    if not state["use_web"] or not web_search_configured():
        return {"web_sources": [], "web_search_error": None}

    try:
        results = web_search(state["question"])
    except WebSearchError as exc:
        return {"web_sources": [], "web_search_error": str(exc)}

    web_sources: list[SourceDict] = [
        {"kind": "web", "title": r.title or r.url, "reference": r.url, "text": r.content}
        for r in results
    ]
    return {"web_sources": web_sources, "web_search_error": None}


def write(state: ResearchState) -> dict:
    numbered = _numbered_sources(state)
    context = _format_sources(numbered)

    user_content = f"Sources:\n\n{context}\n\nQuestion: {state['question']}"
    if state["revision_count"] > 0 and state["review_notes"]:
        user_content += (
            "\n\nA previous draft was reviewed and needs revision. Address every "
            f"point below in this draft:\n{state['review_notes']}"
        )

    messages = [SystemMessage(content=WRITE_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    response = _llm().invoke(messages)
    return {"draft": _text(response.content)}


def review(state: ResearchState) -> dict:
    numbered = _numbered_sources(state)
    context = _format_sources(numbered)

    user_content = f"Sources:\n\n{context}\n\nDraft report:\n\n{state['draft']}"
    messages = [SystemMessage(content=REVIEW_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    response = _llm().invoke(messages)
    text = _text(response.content).strip()

    verified = text.upper().startswith("VERIFIED")
    notes = "" if verified else text.split("\n", 1)[1].strip() if "\n" in text else text
    return {"verified": verified, "review_notes": notes}


def route_after_review(state: ResearchState) -> str:
    if state["verified"]:
        return "finalize"
    if state["revision_count"] >= settings.max_research_revisions:
        return "finalize"
    return "revise"


def bump_revision(state: ResearchState) -> dict:
    return {"revision_count": state["revision_count"] + 1}


def finalize(state: ResearchState) -> dict:
    status = "Verified against sources." if state["verified"] else "Reviewed; some claims may be unsupported."
    footer = f"\n\n---\n**Review:** {status}"
    if not state["verified"] and state["review_notes"]:
        footer += f"\n\n{state['review_notes']}"
    if state["web_search_error"]:
        footer += f"\n\n*Web search was requested but failed: {state['web_search_error']}. Report uses local sources only.*"
    return {"report": state["draft"] + footer}


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("gather", gather)
    graph.add_node("write", write)
    graph.add_node("review", review)
    graph.add_node("revise", bump_revision)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "gather")
    graph.add_edge("gather", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", route_after_review, {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "write")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def run_research(question: str, local_sources: list[SourceDict], use_web: bool) -> ResearchState:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial: ResearchState = {
        "question": question,
        "use_web": use_web,
        "local_sources": local_sources,
        "web_sources": [],
        "web_search_error": None,
        "draft": "",
        "review_notes": "",
        "verified": False,
        "revision_count": 0,
        "report": "",
    }
    return _compiled_graph.invoke(initial)
