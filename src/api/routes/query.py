"""Query endpoint: hybrid retrieval (no generation yet)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_retriever
from src.rag.retrievers.protocol import Retriever
from src.types import Query, RetrievalResult

router = APIRouter()


@router.post("/query", response_model=list[RetrievalResult])
async def query(
    payload: Query, retriever: Retriever = Depends(get_retriever)
) -> list[RetrievalResult]:
    return await retriever.retrieve(payload)
