# AI Research Assistant

Retrieval-augmented document Q&A service — FastAPI + LangChain ingestion,
chunking, and TF-IDF vector retrieval, with LLM grounded synthesis and
cited sources. Ships as one Docker image with its own frontend.

A source document is normalized, split into overlapping passages, and
vectorized into a TF-IDF index (`app/vectorstore.py`). At query time, the
question is vectorized with the same fitted vectorizer, the highest-scoring
passages are retrieved by cosine similarity, and a language model is
prompted with only those passages, required to answer solely from them and
to cite which one it used (`app/synthesis.py`). The language model backend
is pluggable — the default wiring calls a Groq-hosted model through
LangChain, and swapping providers touches only that one module. Without a
model configured, the retrieved passages are returned directly instead of
a synthesized answer, so the retrieval half stands on its own.

The frontend at `/` is served by the same process — ask a question, upload
or paste a document, and watch the architecture diagram trace which
pipeline step is running.

Beyond single-shot Q&A, `/api/research` runs a three-agent LangGraph
pipeline (`app/agents/graph.py`): one call gathers local passages and,
optionally, live web results via Tavily; a second drafts a cited markdown
report; a third independently reviews that draft against its sources and
sends it back for revision if a claim doesn't hold up. Documents can also
be tagged at ingest time and filtered by tag, and every research report
downloads as a `.md` file straight from the browser.

## How it works

```
Source documents ──▶ Chunking ──▶┐
                                  ├─▶ TF-IDF vectorizer ──▶ Vector index ──▶ Retriever ──▶ Synthesis ──▶ grounded, cited response
User query ───────────────────────┘
```

1. **Ingest** — a source file (Markdown, plain text, or PDF) is read and
   normalized into plain text.
2. **Chunk** — normalized text is split into overlapping passages
   (`app/chunking.py`), breaking on paragraph and sentence boundaries
   before falling back to a hard cut.
3. **Vectorize** — every passage is vectorized with a TF-IDF vectorizer
   (`app/vectorstore.py`) fit across the whole corpus. Incoming queries are
   vectorized with the same fitted vectorizer, landing in the same space.
4. **Retrieve** — a cosine-similarity search returns the top-k passages
   above a minimum relevance score.
5. **Synthesize** — a Groq-hosted language model, called through
   LangChain, is prompted with only the retrieved passages and instructed
   to answer solely from them and to cite which passage it used
   (`app/synthesis.py`). If no `GROQ_API_KEY` is configured, the retrieved
   passages are returned directly instead of a synthesized answer — the
   service is still useful without one.

## Research pipeline (`/api/research`)

`/api/query` above is one retrieval, one model call. `/api/research` is a
different, heavier tool: a LangGraph state graph (`app/agents/graph.py`)
that runs three independent model calls per request.

```
gather ──▶ write ──▶ review ──(needs revision, budget left)──▶ write
                                \──(verified, or budget spent)──▶ finalize
```

1. **Gather** — local retrieval runs before the graph is invoked (same
   `app/vectorstore.py` search as `/api/query`); the graph's `gather` node
   adds live web results from Tavily on top, when `use_web` is true and
   `TAVILY_API_KEY` is set. A Tavily failure is caught and noted in the
   final report rather than failing the whole request — local-only
   research still goes through.
2. **Write** — drafts a structured markdown report (a direct answer, a
   "Key findings" section, a "Synthesis" paragraph) citing every claim by
   source number, from local and web sources numbered together.
3. **Review** — a *second, independent* model call re-reads the draft
   against the same numbered sources looking specifically for uncited or
   miscited claims — not the writer grading its own work. If it finds
   problems, its notes go back into another `write` pass; this loops at
   most `MAX_RESEARCH_REVISIONS` times (default 1) before the report ships
   with an honest "needs revision" status rather than looping forever.

The response always includes every source retrieved (local and web) plus
whether the report was verified, so the evidence is inspectable even when
the review didn't fully pass.

## Design notes

**TF-IDF instead of a neural embedding model.** This project targets a
Python version ahead of what PyTorch currently ships wheels for, which
rules out `sentence-transformers` for local, dependency-light embeddings.
Rather than require an embedding API key, retrieval uses scikit-learn's
TF-IDF vectorizer: real vector search, no GPU, no model download, no key.
The vector-store interface doesn't care how a vector was produced
(`app/vectorstore.py`), so a dense embedding model is a one-file swap.

**Retrieval and generation share nothing but the index.** The retriever
has no knowledge of which model reads its output — either half can be
swapped independently.

**Review is a separate model call, not the writer checking its own
work.** A model rarely catches its own citation errors mid-generation.
Asking again, from an explicitly critical framing, with nothing but the
draft and its sources, does — in practice this catches real miscitations
(a claim attributed to a source that doesn't actually support it) that
the writing pass missed.

**In-memory index, rebuilt on startup.** The default deployment has no
persistent disk, so the vector index is rebuilt from `sample_data/` every
time the process starts, and anything ingested at runtime through
`/api/ingest/*` lives only as long as that process. Good enough for a demo
service; a production corpus would move ingestion to a persistent store
(pgvector, a managed vector DB, or a mounted disk running the same Chroma
or FAISS store behind this same interface).

## Running with Docker

```bash
docker build -t ai-research-assistant .
docker run -p 8000:8000 --env-file .env ai-research-assistant
```

Or with Compose, which reads `GROQ_API_KEY` / `GROQ_MODEL` /
`TAVILY_API_KEY` / `APP_PASSWORD` from a `.env` file in the project root
the same way:

```bash
cp .env.example .env   # add your GROQ_API_KEY to enable synthesis
docker compose up --build
```

Either way, open `http://localhost:8000`. The image runs as a non-root
user, exposes a `HEALTHCHECK` against `/health`, and pins its own Python
version (3.12) independent of whatever Python the host has installed — see
the Dockerfile's top comment for why. It's the same image any Docker-based
host (Fly.io, Railway, a VPS, Render's Docker runtime) deploys from
directly.

## Running locally without Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your GROQ_API_KEY to enable synthesis

uvicorn app.main:app --reload
```

Open `http://localhost:8000`. Without a `GROQ_API_KEY`, ingestion,
retrieval, and the frontend all work exactly the same — queries return the
matching passages instead of a synthesized answer.

### Tests

```bash
pytest
```

Tests cover chunking, the vector index, every API route, the Tavily
wrapper, and the LangGraph research pipeline itself (verified-on-first-pass,
needs-revision-then-recovers, and the revision-budget cap) — all with the
language model and web search mocked out, so the suite needs no real API
keys.

## API

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Service status: Groq/Tavily configured, documents and chunks indexed |
| `/api/documents` | GET | List indexed documents, chunk counts, and tags. `?tag=x` filters to one tag |
| `/api/query` | POST | `{"question": str}` → retrieved passages, plus a synthesized answer if configured |
| `/api/research` | POST | `{"question": str, "use_web": bool}` → the gather/write/review pipeline's cited markdown report |
| `/api/ingest/text` | POST | `{"title": str, "text": str, "tags": [str]}` → index pasted text |
| `/api/ingest/file` | POST | Multipart upload (`.md`, `.txt`, `.pdf`, up to 10 MB) + optional `tags` form field (comma-separated) |

Full details in `sample_data/api-reference.md` — which is itself part of
the default indexed corpus, so you can ask the running service about its
own API.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Enables synthesized answers and the research pipeline. Without it, `/api/query` returns retrieved passages only and `/api/research` returns 503. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model used for synthesis and for the research pipeline's write/review steps. |
| `TAVILY_API_KEY` | *(empty)* | Enables the `/api/research` web-search step. Everything else works without it. |
| `WEB_SEARCH_MAX_RESULTS` | `4` | Web results fetched per research query. |
| `MAX_RESEARCH_REVISIONS` | `1` | Cap on the write→review revision loop before a report ships as-is. |
| `APP_PASSWORD` | *(empty)* | Gates every route except `/health` behind HTTP Basic Auth. Set this on any public deployment so a public URL can't burn your Groq/Tavily quota. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Passage size and overlap, in characters. |
| `TOP_K` | `4` | Passages retrieved per query. |
| `MIN_RELEVANCE_SCORE` | `0.05` | Minimum cosine similarity for a passage to count as relevant. |

## Deploying

This ships as one Docker image (see "Running with Docker" above), so any
host that runs a container from a Dockerfile works the same way:

### Render

`render.yaml` is a ready-to-use Blueprint that builds straight from the
`Dockerfile` (`runtime: docker`) rather than Render's native Python
buildpack:

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo.
3. Set `GROQ_API_KEY` (get one at [console.groq.com](https://console.groq.com)).
   Optionally set `TAVILY_API_KEY` (get one at [tavily.com](https://tavily.com))
   to enable web search in `/api/research`, and `APP_PASSWORD` for a public
   deployment.
4. Deploy. Render builds the image from the Dockerfile and runs it with
   its own `$PORT` injected — the image's `CMD` reads that at container
   start rather than assuming a fixed port, and `/health` is the health
   check path.

### Fly.io, Railway, a plain VPS, or anywhere else that runs a Dockerfile

`fly launch` / Railway's "Deploy from Dockerfile" / a bare
`docker run -p 80:8000 --env-file .env $(docker build -q .)` all work
unchanged — nothing in the image is Render-specific beyond reading `$PORT`
when the platform sets one (it falls back to 8000 otherwise).

### What's the same everywhere

No platform in the free/hobby tier here ships a persistent disk by
default, and this service doesn't need one to run — see "In-memory index,
rebuilt on startup" above for what that means for anything ingested at
runtime through `/api/ingest/*`.
