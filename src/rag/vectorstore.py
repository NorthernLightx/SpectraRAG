"""Qdrant vector store wrapper. Supports `:memory:` (in-process) and remote URLs.

The payload schema persists the full Chunk so the corpus can be re-materialised
at API startup without a separate manifest file: `_wire_retriever_from_settings`
(in `src/api/main.py`) calls `scroll_chunks()` to seed BM25 + chunks_by_id.
Older collections written before this schema (payload only `{chunk_id, paper_id}`)
won't round-trip — re-ingest with `scripts/bootstrap_corpus.py --force`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.types import Chunk


@dataclass(frozen=True)
class VectorMatch:
    """A scored vector hit."""

    chunk_id: str
    score: float


def _chunk_to_payload(chunk: Chunk) -> dict[str, object]:
    """Serialise a Chunk into a Qdrant payload dict.

    Optional fields (`section`, `context`) are stored as `None` rather than
    omitted, keeping the schema dense and the round-trip explicit.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "page_numbers": list(chunk.page_numbers),
        "text": chunk.text,
        "section": chunk.section,
        "context": chunk.context,
        "metadata": dict(chunk.metadata),
    }


def _payload_to_chunk(payload: dict[str, object]) -> Chunk | None:
    """Reverse of `_chunk_to_payload`. Returns None on payloads written by an
    older schema (before this column existed) — the caller logs and skips so a
    stale collection degrades to "retriever not wired" rather than crashing."""
    required = ("chunk_id", "paper_id", "page_numbers", "text")
    if not all(key in payload for key in required):
        return None
    page_numbers_raw = payload["page_numbers"]
    if not isinstance(page_numbers_raw, list):
        return None
    metadata_raw = payload.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    section_raw = payload.get("section")
    context_raw = payload.get("context")
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        paper_id=str(payload["paper_id"]),
        page_numbers=[int(n) for n in page_numbers_raw],
        text=str(payload["text"]),
        section=section_raw if isinstance(section_raw, str) else None,
        context=context_raw if isinstance(context_raw, str) else None,
        metadata=metadata,
    )


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
        # Three url forms:
        #   `:memory:`      — in-process, ephemeral. Tests + dev.
        #   `path:/some/dir` — in-process, persistent file store. The deploy
        #                      uses this with a snapshot baked into the
        #                      Docker image so there's no external Qdrant
        #                      service. qdrant-client's local mode supports
        #                      the same hybrid query surface as the remote
        #                      mode (sqlite-backed, sub-ms latency).
        #   anything else   — treated as a remote http(s) URL.
        if url == ":memory:":
            self._client = AsyncQdrantClient(":memory:")
        elif url.startswith("path:"):
            self._client = AsyncQdrantClient(path=url.removeprefix("path:"))
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

    async def delete_collection(self) -> None:
        """Drop the collection if it exists.

        `ensure_collection` is create-if-absent and will NOT clear an existing
        collection, so a ``--force`` re-ingest (scripts/bootstrap_corpus.py) must
        delete first — otherwise it upserts the new corpus on top of the old one
        and leaves stale chunks (e.g. pre-classifier figures) behind.
        """
        existing = await self._client.get_collections()
        if any(c.name == self._collection for c in existing.collections):
            await self._client.delete_collection(collection_name=self._collection)

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
                payload=_chunk_to_payload(chunk),
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

    async def search(
        self, vector: list[float], top_k: int, *, paper_filter: str | None = None
    ) -> list[VectorMatch]:
        """Top-`top_k` dense hits, optionally filtered to one paper.

        `paper_filter` issues a Qdrant payload filter on `paper_id`. Used by
        the eval-side paper-id-aware retrieval path; production callers pass
        `None`. Filtering at the Qdrant layer (rather than post-filtering
        results) keeps the candidate pool size meaningful — without it,
        `top_k=50` returned across 20 papers leaves only ~2.5 same-paper hits
        for rerank on a paper-specific query.
        """
        qdrant_filter: qdrant_models.Filter | None = None
        if paper_filter is not None:
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="paper_id",
                        match=qdrant_models.MatchValue(value=paper_filter),
                    )
                ]
            )
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [
            VectorMatch(chunk_id=str(point.payload["chunk_id"]), score=float(point.score))
            for point in response.points
            if point.payload and "chunk_id" in point.payload
        ]

    async def scroll_chunks(self, *, batch_size: int = 256) -> list[Chunk]:
        """Read every chunk back from the collection's payload — paginated scroll.

        Returns [] if the collection doesn't exist. Skips any payload missing
        required fields (older schema) so a stale collection produces a smaller
        result rather than raising. The API startup path uses this to seed BM25
        + chunks_by_id; tests use it to verify round-trip fidelity.
        """
        existing = await self._client.get_collections()
        if not any(c.name == self._collection for c in existing.collections):
            return []
        chunks: list[Chunk] = []
        offset: qdrant_models.ExtendedPointId | None = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload is None:
                    continue
                chunk = _payload_to_chunk(record.payload)
                if chunk is not None:
                    chunks.append(chunk)
            if offset is None:
                break
        return chunks
