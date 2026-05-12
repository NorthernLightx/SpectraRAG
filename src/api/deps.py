"""FastAPI dependency-injection wiring."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from src.config.settings import Settings, load_settings
from src.observability.langfuse import LangfuseLike
from src.rag.generate import Generator
from src.rag.retrievers.protocol import Retriever
from src.types import Chunk


class _RetrieverState:
    """Module-level holder. The app sets this at startup; tests override via DI."""

    instance: Retriever | None = None


class _GeneratorState:
    instance: Generator | None = None


class _TracerState:
    instance: LangfuseLike | None = None


class _ChunksState:
    """Read-only chunk index keyed by chunk_id. Populated at lifespan startup
    alongside the retriever; used by /figures to enumerate figure-kind chunks
    without re-scrolling Qdrant on each request."""

    instance: dict[str, Chunk] | None = None


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
    _RetrieverState.instance = retriever


def get_generator() -> Generator:
    """Return the configured Generator, or raise 503 if not wired."""
    if _GeneratorState.instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generator not configured. Configure an LLM client before requesting answers.",
        )
    return _GeneratorState.instance


def set_generator(generator: Generator) -> None:
    _GeneratorState.instance = generator


def get_tracer() -> LangfuseLike | None:
    """Return the configured Langfuse tracer or None (no-op tracing)."""
    return _TracerState.instance


def set_tracer(tracer: LangfuseLike | None) -> None:
    _TracerState.instance = tracer


def get_chunks() -> dict[str, Chunk]:
    """Return the loaded chunk index, or raise 503 if no corpus is loaded."""
    if _ChunksState.instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Corpus not loaded. Ingest a corpus before listing figures.",
        )
    return _ChunksState.instance


def set_chunks(chunks: dict[str, Chunk]) -> None:
    _ChunksState.instance = chunks
