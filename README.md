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

Or with Compose, which reads `GROQ_API_KEY` / `GROQ_MODEL` / `APP_PASSWORD`
from a `.env` file in the project root the same way:

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

Tests cover chunking, the vector index, and every API route — including
the grounded-answer path, via a monkeypatched language model call, so the
suite needs no real API key.

## API

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Service status, always unauthenticated |
| `/api/documents` | GET | List indexed documents and chunk counts |
| `/api/query` | POST | `{"question": str}` → retrieved passages, plus a synthesized answer if configured |
| `/api/ingest/text` | POST | `{"title": str, "text": str}` → index pasted text |
| `/api/ingest/file` | POST | Multipart upload (`.md`, `.txt`, `.pdf`, up to 10 MB) |

Full details in `sample_data/api-reference.md` — which is itself part of
the default indexed corpus, so you can ask the running service about its
own API.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Enables synthesized answers. Without it, `/api/query` returns retrieved passages only. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model used for synthesis. |
| `APP_PASSWORD` | *(empty)* | Gates every route except `/health` behind HTTP Basic Auth. Set this on any public deployment so a public URL can't burn your Groq quota. |
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
3. Set `GROQ_API_KEY` (get one at [console.groq.com](https://console.groq.com))
   and, for a public deployment, `APP_PASSWORD`.
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
