"""FastAPI dependency-injection wiring."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from src.config.settings import Settings, load_settings
from src.rag.retrievers.protocol import Retriever


class _RetrieverState:
    """Module-level holder. The app sets this at startup; tests override via DI."""

    instance: Retriever | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor for FastAPI dependencies."""
    return load_settings()


def get_retriever() -> Retriever:
    """Return the configured Retriever, or raise 503 if no corpus is loaded."""
    if _RetrieverState.instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever not configured. Ingest a corpus before querying.",
        )
    return _RetrieverState.instance


def set_retriever(retriever: Retriever) -> None:
    """Install the active retriever for the running app."""
    _RetrieverState.instance = retriever
