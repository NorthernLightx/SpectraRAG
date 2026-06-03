"""Experimental DCI retrieval: an LLM agent greps the corpus (opt-in, key-bound).

Off by default (`RAG_ENABLE_DCI`). The agent runs server-side, so it needs an
OpenRouter key: the server's own when configured, else the caller's via the
`X-OpenRouter-Key` header. Unlike normal generation (which goes browser-direct so
the server never sees the key), this mode does receive the key — used in-memory for
the request, never logged or stored. The key is read from a header, not the request
body, so it never lands in the request log.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from src.api.deps import get_chunks, get_settings
from src.config.settings import Settings
from src.dci.tools import CorpusTools
from src.llm.openrouter import OpenRouterClient
from src.observability.logging import get_logger
from src.rag.retrievers.dci import DciRetriever, build_dci_corpus
from src.types import Chunk, Query, RetrievalResponse

_log = get_logger(__name__)
router = APIRouter()


class _DciCorpusState:
    """The grep-able corpus, built once from the static chunk index and reused."""

    tools: CorpusTools | None = None
    sur_to_chunk: dict[str, str] | None = None


@router.post("/query/dci", response_model=RetrievalResponse)
async def query_dci(
    payload: Query,
    x_openrouter_key: str | None = Header(default=None, alias="X-OpenRouter-Key"),
    chunks: dict[str, Chunk] = Depends(get_chunks),
    settings: Settings = Depends(get_settings),
) -> RetrievalResponse:
    if not settings.enable_dci:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "DCI retrieval is disabled (set RAG_ENABLE_DCI)."
        )
    server_key = (
        settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else None
    )
    key = (x_openrouter_key or server_key or "").strip()
    if not key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "DCI runs server-side and needs an OpenRouter key. The server has none, so pass yours via "
            "the X-OpenRouter-Key header (used in-memory for this request, not stored).",
        )
    if _DciCorpusState.tools is None:
        _DciCorpusState.tools, _DciCorpusState.sur_to_chunk = build_dci_corpus(chunks)
    retriever = DciRetriever(
        _DciCorpusState.tools,
        _DciCorpusState.sur_to_chunk or {},
        chunks,
        OpenRouterClient(api_key=key),
        settings.dci_model,
    )
    results = await retriever.retrieve(payload)
    # Log the query and the key SOURCE, never the key value.
    _log.info(
        "dci.query",
        query=payload.text,
        returned=len(results),
        key_source="user" if x_openrouter_key else "server",
    )
    return RetrievalResponse(results=results, routing=None)
