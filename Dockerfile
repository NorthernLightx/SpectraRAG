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
# Page PNGs for the multi-modal vision-generation path. The browser sends
# these URLs to OpenRouter as image content blocks so a vision-capable model
# (gpt-4o, claude, qwen3-vl) sees the actual page pixels. Render with
# `python -m scripts.render_pages --pdf-dir data/papers` before `docker
# build`; the StaticFiles mount in src/api/main.py guards on existence so
# `data/pages` being absent doesn't break the build (the deploy just serves
# text-only retrieval).
COPY --chown=app:app data/pages /home/app/data/pages
# Human-readable paper titles (data/paper_titles.json); /papers falls back to
# paper_id when absent.
COPY --chown=app:app data/paper_titles.json /home/app/data/paper_titles.json

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

# Pre-download model weights into the HuggingFace cache so no request pays an
# HF fetch at runtime. Two models load on different paths: the bge-m3 embedder
# during startup wiring, and the reranker cross-encoder (src/rag/rerank.py)
# lazily on the first /query. The deploy reranks on CPU (no GPU on Cloud Run),
# where the 568M bge-reranker-v2-m3 costs minutes per query, so this image bakes
# the small MiniLM cross-encoder the `cpu` profile names (a unit test keeps the
# two in step). Cache lives at /home/app/.cache/huggingface/ (default for `app`).
RUN /home/app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ADR 0028: bake the ColQwen2 visual encoder so the multimodal serve path pays
# no HuggingFace fetch at startup — a ~4 GB cold download would blow the Cloud
# Run startup window. from_pretrained caches the adapter, its Qwen2-VL-2B base,
# and the processor. Only the query is encoded at serve time; the page vectors
# are pre-built into the Qdrant snapshot (scripts/build_visual_index.py). Adds
# ~4 GB to the image; the visual leg only activates when RAG_ENABLE_MULTIMODAL
# is set AND the visual collection is populated.
RUN /home/app/.venv/bin/python -c \
    "from colpali_engine.models import ColQwen2, ColQwen2Processor; ColQwen2.from_pretrained('vidore/colqwen2-v1.0'); ColQwen2Processor.from_pretrained('vidore/colqwen2-v1.0')"

# RAG_PROFILE=cpu: the retrieval stack lives in src/config/profiles/cpu.yaml so
# `eval_run --profile cpu` measures exactly what this image serves.
ENV RAG_PROFILE=cpu \
    RAG_PAGES_DIR=/home/app/data/pages \
    RAG_QDRANT_URL=path:/home/app/qdrant_local

ENV PATH="/home/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_ENV=prod \
    RAG_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
