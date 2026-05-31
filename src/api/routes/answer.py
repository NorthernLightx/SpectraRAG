"""Answer endpoint: retrieve + generate, with Langfuse + OTel instrumentation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.deps import get_generator, get_retriever, get_settings, get_tracer
from src.api.rate_limit import limiter
from src.config.settings import Settings
from src.observability.langfuse import LangfuseLike, trace_query
from src.observability.otel import get_tracer as get_otel_tracer
from src.rag.generate import Generator
from src.rag.page_budget import resolve_whole_doc_pages
from src.rag.retrievers.protocol import Retriever
from src.types import Answer, Query

router = APIRouter()


# slowapi keys by `request.client.host`, so the `request: Request` parameter
# is mandatory for the decorator to extract the IP. 10/minute is a loose
# ceiling sized for a low-traffic demo. Tests reset the limiter between cases
# so the limit doesn't leak.
@router.post("/answer", response_model=Answer)
@limiter.limit("10/minute")
async def answer(
    request: Request,
    payload: Query,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    tracer: LangfuseLike | None = Depends(get_tracer),
    settings: Settings = Depends(get_settings),
) -> Answer:
    otel_tracer = get_otel_tracer()
    with otel_tracer.start_as_current_span("rag.retrieve") as span:
        span.set_attribute("rag.query.text_len", len(payload.text))
        span.set_attribute("rag.query.top_k", payload.top_k)
        # ADR 0024 route-by-fit: when the query names a paper and that document
        # fits the page budget, feed the WHOLE document's page images instead of
        # the top-k RAG cut. Decision inputs (paper id, budget, on-disk page
        # count) need no retrieval, so a whole-doc hit skips retrieval entirely.
        # Falls back to RAG when route-by-fit is off, the query isn't
        # paper-scoped, or the doc is missing / over budget.
        whole_doc = None
        paper_id = payload.paper_id_filter()
        if settings.page_budget is not None and settings.pages_dir is not None and paper_id:
            whole_doc = resolve_whole_doc_pages(paper_id, settings.pages_dir, settings.page_budget)
        if whole_doc is not None:
            retrieved = whole_doc
            span.set_attribute("rag.route_by_fit", "whole_doc")
            span.set_attribute("rag.route_by_fit.pages", len(whole_doc))
        else:
            retrieved = await retriever.retrieve(payload)
            span.set_attribute("rag.route_by_fit", "rag")
        span.set_attribute("rag.retrieved.count", len(retrieved))

    with otel_tracer.start_as_current_span("rag.generate") as span:
        span.set_attribute("rag.context.chunks", len(retrieved))
        result = await generator.answer(payload.text, retrieved)
        span.set_attribute("rag.tokens.in", result.tokens_in)
        span.set_attribute("rag.tokens.out", result.tokens_out)
        span.set_attribute("rag.citations.count", len(result.citations))

    trace_query(tracer, query=payload, retrieved=retrieved, answer=result)
    # The Generator doesn't know about the retrieval results it was handed
    # (it only consumes them to build context). The bundled web UI wants to
    # show "what the LLM saw" — the route layer attaches the list here so
    # downstream eval / unit-test paths that construct Answer directly stay
    # backwards-compatible (default `[]`).
    return result.model_copy(update={"retrieved": retrieved})
