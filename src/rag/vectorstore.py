"""Qdrant vector store wrapper. Supports `:memory:` (in-process) and remote URLs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.types import Chunk


@dataclass(frozen=True)
class VectorMatch:
    """A scored vector hit."""

    chunk_id: str
    score: float


class QdrantVectorStore:
    """Async wrapper over qdrant-client. Handles collection lifecycle + upsert + search."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        dim: int,
        *,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self._collection = collection_name
        self._dim = dim
        self._distance = distance
        if url == ":memory:":
            self._client = AsyncQdrantClient(":memory:")
        else:
            self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        existing = await self._client.get_collections()
        if any(c.name == self._collection for c in existing.collections):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dim, distance=self._distance),
        )

    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks/vectors length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )
        if not chunks:
            return
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id)),
                vector=vector,
                payload={"chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def count(self) -> int:
        """Return the number of points in the collection; 0 if it doesn't exist.

        Used by `scripts/bootstrap_corpus.py` for idempotent re-runs — a populated
        collection means ingestion already happened, skip unless --force.
        """
        existing = await self._client.get_collections()
        if not any(c.name == self._collection for c in existing.collections):
            return 0
        result = await self._client.count(collection_name=self._collection, exact=True)
        return int(result.count)

    async def search(self, vector: list[float], top_k: int) -> list[VectorMatch]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
        )
        return [
            VectorMatch(chunk_id=str(point.payload["chunk_id"]), score=float(point.score))
            for point in response.points
            if point.payload and "chunk_id" in point.payload
        ]
