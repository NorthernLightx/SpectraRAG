"""FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import health, query


def create_app() -> FastAPI:
    """Construct the FastAPI app. Factory shape keeps test setup clean."""
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
    return app


app = create_app()
