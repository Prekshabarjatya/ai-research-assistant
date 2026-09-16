"""Grounded answer synthesis over retrieved passages.

The model only ever sees the passages the retriever picked, numbered, in
the prompt — never the raw corpus, and never a chance to answer from
whatever it already knows about the topic. It's told outright to say so
when the passages don't cover the question, rather than fill the gap from
parametric memory.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.vectorstore import SearchHit

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question using ONLY the \
numbered passages given below. Follow these rules strictly:

1. Base your answer only on the provided passages. Do not use outside knowledge.
2. If the passages don't contain enough information to answer, say so plainly \
instead of guessing.
3. After the answer, cite which passage number(s) it came from, like "(source: 2)".
4. Be concise. Answer in prose, not a list, unless the question asks for a list."""


class SynthesisError(RuntimeError):
    """Raised when the LLM call itself fails (bad key, rate limit, network)."""


def is_configured() -> bool:
    return bool(settings.groq_api_key)


def _build_context(sources: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{i}] (from {s.document}, chunk {s.chunk_index})\n{s.text}"
        for i, s in enumerate(sources, start=1)
    )


def synthesize(question: str, sources: list[SearchHit]) -> str:
    if not is_configured():
        raise SynthesisError("GROQ_API_KEY is not configured.")

    llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.1)
    context = _build_context(sources)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Passages:\n\n{context}\n\nQuestion: {question}"),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # network error, bad key, rate limit, etc.
        raise SynthesisError(str(exc)) from exc

    content = response.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content.strip()
