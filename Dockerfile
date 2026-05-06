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

ENV PATH="/home/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_ENV=prod \
    RAG_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
