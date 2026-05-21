# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.5.4-python3.12-bookworm-slim AS builder

WORKDIR /app

# Install only main deps; dev deps aren't needed in the runtime image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source and install the project itself (wheel build).
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

# OS deps for psycopg, pymupdf, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libpq5 \
       libgomp1 \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /home/app

# Copy the virtualenv from the builder stage.
COPY --from=builder --chown=app:app /app/.venv /home/app/.venv
COPY --from=builder --chown=app:app /app/src /home/app/src
# Static UI bundled into the image — FastAPI mounts this at "/" so the same
# container serves both the API and the demo page. Skipped if the path is
# absent (the StaticFiles mount in src/api/main.py guards on existence).
COPY --chown=app:app web /home/app/web
# Page PNGs for the multi-modal vision-generation path. The browser sends
# these URLs to OpenRouter as image content blocks so a vision-capable model
# (gpt-4o, claude, qwen3-vl) sees the actual page pixels. Render with
# `python -m scripts.render_pages --pdf-dir data/papers` before `docker
# build`; the StaticFiles mount in src/api/main.py guards on existence so
# `data/pages` being absent doesn't break the build (the deploy just serves
# text-only retrieval).
COPY --chown=app:app data/pages /home/app/data/pages

# Baked-in Qdrant snapshot. `qdrant-client` runs in embedded mode against
# this directory (`url='path:/home/app/qdrant_local'`), so the deploy needs
# no external Qdrant — the entire vector index ships inside the image.
# Build it before `docker build`:
#   uv run python -m scripts.bootstrap_corpus \
#       --pdf-dir data/papers \
#       --qdrant path:./qdrant_local \
#       --ollama http://localhost:11434
# When the directory is empty (only .gitkeep), the lifespan handler logs
# `skip_empty_corpus` and /answer returns 503 — same fallback as a missing
# pages_dir. Re-baking is idempotent in the source script.
COPY --chown=app:app qdrant_local /home/app/qdrant_local

# Pre-download the model weights into the HuggingFace cache so no request
# pays a multi-GB HF fetch at runtime. Two models load on different paths:
# the bge-m3 embedder during startup wiring, and the bge-reranker-v2-m3
# cross-encoder (src/rag/rerank.py) lazily on the first /query. Baking both
# keeps that fetch out of the request path. Cache lives at
# /home/app/.cache/huggingface/ (default location for the `app` user).
# Skipping this just means the first request after a cold start downloads.
RUN /home/app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

ENV RAG_EMBEDDER_BACKEND=sentence_transformers \
    RAG_PAGES_DIR=/home/app/data/pages \
    RAG_QDRANT_URL=path:/home/app/qdrant_local

ENV PATH="/home/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_ENV=prod \
    RAG_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
