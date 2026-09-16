"""FastAPI surface for the retrieval-augmented Q&A service.

One process serves both the JSON API (/health, /api/*) and the static
frontend (everything else, from app/static/) — no separate frontend
deployment or CORS configuration needed.

The vector index lives in process memory (see app/vectorstore.py) and is
rebuilt from `sample_data/` on every startup. Anything ingested at runtime
through /api/ingest/* lives only as long as this process — there's no
persistent disk in the default deployment. See the README's "Design notes"
for what that trades off and where a persistent store would plug in.
"""

import base64
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.ingestion import SUPPORTED_EXTENSIONS, extract_text, load_sample_corpus
from app.schemas import (
    DocumentSummary,
    DocumentsResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    Source,
)
from app.synthesis import SynthesisError, is_configured, synthesize
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

store = VectorStore()


def _load_sample_corpus() -> None:
    for name, text in load_sample_corpus(SAMPLE_DATA_DIR):
        added = store.add_document(name, text)
        logger.info("Indexed %s (%d chunks)", name, added)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_sample_corpus()
    yield


app = FastAPI(title="AI Research Assistant", version="0.1.0", lifespan=lifespan)


# Optional shared-password gate (HTTP Basic) for public deployments. Off by
# default for local dev (settings.app_password == ""). /health is always
# left open so a hosting platform's health check doesn't need credentials.
# Username is ignored on purpose — this is a single shared password, not a
# per-user account system.
_UNPROTECTED_PATHS = {"/health"}


@app.middleware("http")
async def require_shared_password(request: Request, call_next):
    if not settings.app_password or request.url.path in _UNPROTECTED_PATHS:
        return await call_next(request)

    header = request.headers.get("authorization", "")
    supplied = ""
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[len("Basic ") :]).decode("utf-8")
            _, _, supplied = decoded.partition(":")
        except (ValueError, UnicodeDecodeError):
            supplied = ""

    if secrets.compare_digest(supplied, settings.app_password):
        return await call_next(request)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="AI Research Assistant"'},
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "groq_configured": is_configured(),
        "documents_indexed": len(store.document_names),
        "chunks_indexed": store.chunk_count,
    }


@app.get("/api/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    summary = store.document_summary()
    return DocumentsResponse(
        documents=[DocumentSummary(document=name, chunks=n) for name, n in summary],
        total_chunks=store.chunk_count,
    )


@app.post("/api/ingest/text", response_model=IngestResponse)
def ingest_text(payload: IngestTextRequest) -> IngestResponse:
    added = store.add_document(payload.title, payload.text)
    if added == 0:
        raise HTTPException(422, "No text left to index after normalization.")
    return IngestResponse(
        document=payload.title,
        chunks_added=added,
        total_documents=len(store.document_names),
        total_chunks=store.chunk_count,
    )


@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415, f"Unsupported file type {suffix!r}. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB limit.")

    try:
        text = extract_text(file.filename or "upload", raw)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc

    added = store.add_document(file.filename or "upload", text)
    if added == 0:
        raise HTTPException(422, "No extractable text found in that file.")
    return IngestResponse(
        document=file.filename or "upload",
        chunks_added=added,
        total_documents=len(store.document_names),
        total_chunks=store.chunk_count,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    k = payload.top_k or settings.top_k
    hits = store.search(payload.question, k)
    hits = [h for h in hits if h.score >= settings.min_relevance_score]

    sources = [
        Source(document=h.document, chunk_index=h.chunk_index, score=h.score, text=h.text)
        for h in hits
    ]

    if not hits:
        return QueryResponse(
            question=payload.question,
            answer=None,
            grounded=False,
            note="No passage in the indexed corpus is relevant enough to answer this. "
            "Try rephrasing, or ingest a document that covers it.",
            sources=[],
        )

    if not is_configured():
        return QueryResponse(
            question=payload.question,
            answer=None,
            grounded=False,
            note="GROQ_API_KEY is not configured on this deployment, so the closest "
            "matching passages are shown instead of a synthesized answer.",
            sources=sources,
        )

    try:
        answer = synthesize(payload.question, hits)
    except SynthesisError as exc:
        raise HTTPException(502, f"The language model backend failed: {exc}") from exc

    return QueryResponse(
        question=payload.question,
        answer=answer,
        grounded=True,
        note=None,
        sources=sources,
    )


# Registered last: everything not matched by an API route above falls
# through to the static frontend.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
