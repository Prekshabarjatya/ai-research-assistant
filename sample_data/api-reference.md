# API reference

The service exposes a small JSON API alongside the static frontend it
serves at `/`. All endpoints are same-origin, so no CORS configuration is
needed for the bundled frontend.

## GET /health

Returns service status: whether a language model backend is configured,
and how many documents and chunks are currently indexed. Always
unauthenticated, even when a shared password is set, so a hosting
platform's health check doesn't need credentials.

## GET /api/documents

Lists every document currently indexed and how many chunks each
contributed, plus the total chunk count across the whole index.

## POST /api/ingest/text

Body: `{"title": string, "text": string}`. Chunks the given text and adds
it to the index under `title`. Returns how many chunks were added and the
new totals.

## POST /api/ingest/file

Multipart file upload. Accepts `.md`, `.markdown`, `.txt`, and `.pdf`.
Extracts text, chunks it, and adds it to the index the same way as
`/api/ingest/text`. Files over the configured size limit (10 MB by
default) are rejected with a 413.

## POST /api/query

Body: `{"question": string, "top_k": integer, optional}`. Retrieves the
most relevant passages for the question and, if a language model backend
is configured, synthesizes a grounded answer from them. The response
always includes the retrieved `sources` — each with its document name,
chunk index, relevance score, and text — regardless of whether synthesis
ran, so the underlying evidence is visible either way.

If no passage clears the minimum relevance threshold, `answer` is `null`
and `sources` is empty rather than forcing a guess. If retrieval succeeds
but no language model backend is configured, `answer` is `null` and
`sources` are returned directly so the passages are still useful without a
synthesized answer on top.
