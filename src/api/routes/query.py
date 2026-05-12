"""Query endpoint: hybrid retrieval (no generation yet)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_retriever
from src.rag.retrievers.protocol import Retriever
from src.rag.retrievers.routing import get_last_routing_info
from src.types import Query, RetrievalResponse

router = APIRouter()


@router.post("/query", response_model=RetrievalResponse)
async def query(
    payload: Query, retriever: Retriever = Depends(get_retriever)
) -> RetrievalResponse:
    results = await retriever.retrieve(payload)
    # When the retriever is a RoutingRetriever, it has populated the contextvar
    # with its decision; PipelineRetriever wired directly leaves it at None.
    return RetrievalResponse(results=results, routing=get_last_routing_info())
