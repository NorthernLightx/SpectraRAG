"""Retriever Protocol: the only retrieval interface upstream code depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.types import Query, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Retrieve ranked candidates for a query. Source-agnostic."""

    async def retrieve(self, query: Query) -> list[RetrievalResult]: ...
