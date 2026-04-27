"""Minimal Langfuse trace wiring. No-op when keys are not configured."""

from __future__ import annotations

import os
from typing import Any, Protocol, cast, runtime_checkable

from src.types import Answer, Query, RetrievalResult


@runtime_checkable
class LangfuseLike(Protocol):
    """Subset of the Langfuse SDK we depend on. Keeps the seam mockable."""

    def trace(self, *, name: str, input: dict[str, Any], output: dict[str, Any]) -> Any: ...

    def flush(self) -> None: ...


def make_langfuse_client(
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> LangfuseLike | None:
    """Construct a real Langfuse client if all three keys are set; else return None."""
    pk = (
        public_key
        or os.environ.get("LANGFUSE_PUBLIC_KEY")
        or os.environ.get("RAG_LANGFUSE_PUBLIC_KEY", "")
    )
    sk = (
        secret_key
        or os.environ.get("LANGFUSE_SECRET_KEY")
        or os.environ.get("RAG_LANGFUSE_SECRET_KEY", "")
    )
    h = host or os.environ.get("LANGFUSE_HOST") or os.environ.get("RAG_LANGFUSE_HOST", "")
    if not (pk and sk and h):
        return None
    from langfuse import Langfuse

    # Langfuse's SDK trace() has a richer signature; we duck-type via LangfuseLike.
    return cast(LangfuseLike, Langfuse(public_key=pk, secret_key=sk, host=h))


def trace_query(
    client: LangfuseLike | None,
    *,
    query: Query,
    retrieved: list[RetrievalResult],
    answer: Answer | None,
) -> None:
    """Fire a single Langfuse trace summarising the retrieval+generation. No-op if not configured."""
    if client is None:
        return
    output: dict[str, Any] = {
        "retrieved_chunk_ids": [r.chunk_id for r in retrieved],
        "n_retrieved": len(retrieved),
    }
    if answer is not None:
        output.update(
            {
                "answer_text": answer.text,
                "answer_model": answer.model,
                "prompt_version": answer.prompt_version,
                "tokens_in": answer.tokens_in,
                "tokens_out": answer.tokens_out,
                "latency_ms": answer.latency_ms,
                "cited_chunk_ids": [c.chunk_id for c in answer.citations],
            }
        )
    client.trace(
        name="rag.query",
        input={"query": query.text, "top_k": query.top_k},
        output=output,
    )
    client.flush()
