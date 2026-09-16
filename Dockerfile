# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# python:3.12 rather than 3.14 here on purpose: this pins the pytorch-free
# TF-IDF stack (see README "Design notes") to a version PyPI wheels reliably
# cover, independent of whatever Python the host machine happens to run.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
COPY sample_data ./sample_data

RUN pip install --no-cache-dir .

# Runs as a non-root user in the shipped image.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV PORT=8000
EXPOSE 8000

# Shell form deliberately: $PORT needs expansion, and a hosting platform
# that injects its own PORT (Render, and most PaaS Docker runtimes) expects
# the container to listen on it rather than a value baked in at build time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
