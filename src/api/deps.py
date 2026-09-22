"""FastAPI dependency-injection wiring."""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from fastapi import HTTPException, status

from src.config.settings import Settings, load_settings
from src.embeddings.protocol import Embedder
from src.observability.langfuse import LangfuseLike
from src.rag.bm25 import Bm25Index
from src.rag.context import FigureRef, figure_refs
from src.rag.generate import Generator
from src.rag.retrieval_config import RetrievalConfig
from src.rag.retrievers.protocol import Retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk


class _RetrieverState:
    """Module-level holder. The app sets this at startup; tests override via DI."""

    instance: Retriever | None = None


class _GeneratorState:
    instance: Generator | None = None


class _RetrievalConfigState:
    """The retrieval config that was actually wired (text-only when the visual
    leg failed to load), reported on /health."""

    instance: RetrievalConfig | None = None


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


def set_retrieval_config(config: RetrievalConfig) -> None:
    _RetrievalConfigState.instance = config


def peek_retrieval_config() -> RetrievalConfig | None:
    return _RetrievalConfigState.instance


def peek_retriever() -> Retriever | None:
    """Non-raising reader for capability checks such as /health flags."""
    return _RetrieverState.instance


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


class _FiguresState:
    """Figure refs derived from the chunk index, rebuilt when POST /ingest has
    grown it."""

    n_chunks = -1
    refs: ClassVar[list[FigureRef]] = []


def peek_figures() -> list[FigureRef]:
    """Figure and table chunks a question can name, or [] with no corpus."""
    chunks = _ChunksState.instance
    if chunks is None:
        return []
    if len(chunks) != _FiguresState.n_chunks:
        _FiguresState.refs = figure_refs(list(chunks.values()))
        _FiguresState.n_chunks = len(chunks)
    return _FiguresState.refs


class _CorpusHandles:
    """Live retrieval-index handles, set at wiring. ``POST /ingest`` (ADR 0029)
    appends a document at runtime through these same objects (the Bm25Index +
    Qdrant store + embedder the wired retriever reads), so an upsert is visible
    to /query without a restart."""

    embedder: Embedder | None = None
    vectorstore: QdrantVectorStore | None = None
    bm25: Bm25Index | None = None


def set_corpus_handles(embedder: Embedder, vectorstore: QdrantVectorStore, bm25: Bm25Index) -> None:
    _CorpusHandles.embedder = embedder
    _CorpusHandles.vectorstore = vectorstore
    _CorpusHandles.bm25 = bm25


def get_corpus_handles() -> tuple[Embedder, QdrantVectorStore, Bm25Index]:
    """Return the live (embedder, vectorstore, bm25) handles, or raise 503."""
    if (
        _CorpusHandles.embedder is None
        or _CorpusHandles.vectorstore is None
        or _CorpusHandles.bm25 is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Corpus not loaded. Ingest a corpus before uploading documents.",
        )
    return _CorpusHandles.embedder, _CorpusHandles.vectorstore, _CorpusHandles.bm25
