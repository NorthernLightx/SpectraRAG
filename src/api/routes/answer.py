"""Answer endpoint: retrieve + generate, with Langfuse + OTel instrumentation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.deps import get_generator, get_retriever, get_tracer
from src.api.rate_limit import limiter
from src.observability.langfuse import LangfuseLike, trace_query
from src.observability.otel import get_tracer as get_otel_tracer
from src.rag.generate import Generator
from src.rag.retrievers.protocol import Retriever
from src.types import Answer, Query

router = APIRouter()


# Phase 2.2 — slowapi keys by `request.client.host`, so the `request: Request`
# parameter is mandatory for the decorator to extract the IP. 10/minute is a
# loose ceiling for a portfolio-demo backend; tune up or down as the demo
# matures. Tests reset the limiter between cases so the limit doesn't leak.
@router.post("/answer", response_model=Answer)
@limiter.limit("10/minute")
async def answer(
    request: Request,
    payload: Query,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    tracer: LangfuseLike | None = Depends(get_tracer),
) -> Answer:
    otel_tracer = get_otel_tracer()
    with otel_tracer.start_as_current_span("rag.retrieve") as span:
        span.set_attribute("rag.query.text_len", len(payload.text))
        span.set_attribute("rag.query.top_k", payload.top_k)
        retrieved = await retriever.retrieve(payload)
        span.set_attribute("rag.retrieved.count", len(retrieved))

    with otel_tracer.start_as_current_span("rag.generate") as span:
        span.set_attribute("rag.context.chunks", len(retrieved))
        result = await generator.answer(payload.text, retrieved)
        span.set_attribute("rag.tokens.in", result.tokens_in)
        span.set_attribute("rag.tokens.out", result.tokens_out)
        span.set_attribute("rag.citations.count", len(result.citations))

    trace_query(tracer, query=payload, retrieved=retrieved, answer=result)
    return result
