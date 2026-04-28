"""FastAPI app factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from src.api.routes import answer, health, query
from src.config.settings import load_settings
from src.observability.logging import configure_logging, get_logger


def create_app(*, log_file: Path | None = Path("logs/api.log")) -> FastAPI:
    """Construct the FastAPI app. Factory shape keeps test setup clean.

    Logging is configured here so unit tests can pass `log_file=None` to keep
    side-effects out of the test directory.
    """
    settings = load_settings()
    configure_logging(level=settings.log_level, env=settings.env, log_file=log_file)
    log = get_logger(__name__)
    log.info("api.startup", env=settings.env, log_level=settings.log_level)

    app = FastAPI(
        title="Multi-modal Paper RAG",
        version="0.1.0",
        description="RAG over scientific papers comparing pipeline vs visual retrieval.",
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "Multi-modal Paper RAG", "docs": "/docs"}

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(answer.router)
    return app


app = create_app()
