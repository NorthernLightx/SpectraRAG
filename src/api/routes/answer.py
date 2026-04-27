"""Answer endpoint: retrieve + generate, with optional Langfuse trace."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_generator, get_retriever, get_tracer
from src.observability.langfuse import LangfuseLike, trace_query
from src.rag.generate import Generator
from src.rag.retrievers.protocol import Retriever
from src.types import Answer, Query

router = APIRouter()


@router.post("/answer", response_model=Answer)
async def answer(
    payload: Query,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    tracer: LangfuseLike | None = Depends(get_tracer),
) -> Answer:
    retrieved = await retriever.retrieve(payload)
    result = await generator.answer(payload.text, retrieved)
    trace_query(tracer, query=payload, retrieved=retrieved, answer=result)
    return result
