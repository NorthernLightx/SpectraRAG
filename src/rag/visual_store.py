"""Persisted ColQwen2 page index in a Qdrant multivector collection.

The in-memory ``VisualRetriever`` (``src/rag/retrievers/visual.py``) embeds every
page at startup, which needs a GPU; on CPU that startup encode runs for tens of
minutes and blows the Cloud Run startup timeout. This store persists the page
multivectors once, offline, on a GPU (``scripts/build_visual_index.py``) into a
Qdrant multivector collection, so the deploy loads them instead of re-encoding.
At serve time only the query is encoded, and Qdrant computes the MaxSim late-
interaction score natively (qdrant-client 1.10+ multivector, MAX_SIM comparator;
embedded ``path:`` mode supports it as of 1.17).

Kept separate from ``QdrantVectorStore`` (the single-vector bge-m3 text store) so
the baked text collection schema is untouched: this collection is page-granular,
multivector, and sized to the visual encoder's per-token dim (128 for ColQwen2)
where the text one is chunk-granular, single-vector, and 1024-dim. Both live in
the same embedded ``qdrant_local`` directory. ADR 0028.
"""

from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    SearchParams,
    VectorParams,
)

from src.types import RetrievalResult

# Page chunk-id format mirrors the in-memory visual leg and the text page-id so
# RoutingRetriever's page-level RRF fusion merges both legs on the same page
# (src/rag/retrievers/routing.py `_to_page_id`).
_PAGE_CHUNK_FMT = "{paper_id}::p{page_no}::page"


class QdrantVisualStore:
    """Async wrapper over a Qdrant multivector collection of page embeddings.

    Construction mirrors ``QdrantVectorStore``'s three url forms so the same
    ``RAG_QDRANT_URL`` works for both: ``:memory:`` (tests), ``path:/dir``
    (embedded, the deploy), or a remote http(s) URL.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        *,
        dim: int | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._collection = collection_name
        # The encoder's per-token dim; only collection creation reads it, so the
        # serve and eval paths, which query an existing collection, omit it.
        self._dim = dim
        # Embedded path-mode allows one client per on-disk path per process, and
        # the serve path already holds one open for the text store. So the serve
        # wiring passes that shared client in; only an owned client (the offline
        # build, which has the path to itself) is created, and closed, here.
        self._owns_client = client is None
        if client is not None:
            self._client = client
        elif url == ":memory:":
            self._client = AsyncQdrantClient(":memory:")
        elif url.startswith("path:"):
            self._client = AsyncQdrantClient(path=url.removeprefix("path:"))
        else:
            self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        """Create the multivector collection if absent (create-if-missing).

        MAX_SIM is the ColBERT/ColPali late-interaction comparator: for each
        query token, take the max similarity over the page's patch vectors, then
        sum across query tokens. The base distance is DOT, not COSINE, to match
        colpali's ``score_multi_vector`` exactly, which scores raw dot products
        (``einsum(...).max().sum()``) with no per-vector normalization, so COSINE
        would diverge from the offline eval's ranking on non-unit vectors.

        ``on_disk`` and ``m=0`` only change a Qdrant server; embedded mode keeps
        vectors in memory and scores every page regardless. A server fixes
        ``on_disk`` at creation, and without it holds every page vector in RAM.
        ``m=0`` skips building an HNSW graph that exact search never reads.
        """
        if self._dim is None:
            raise ValueError(
                f"creating {self._collection!r} needs the visual encoder's per-token dim"
            )
        existing = await self._client.get_collections()
        if any(c.name == self._collection for c in existing.collections):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._dim,
                distance=Distance.DOT,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                on_disk=True,
                hnsw_config=HnswConfigDiff(m=0),
            ),
        )

    async def delete_collection(self) -> None:
        """Drop the collection if it exists (for ``--force`` re-builds)."""
        existing = await self._client.get_collections()
        if any(c.name == self._collection for c in existing.collections):
            await self._client.delete_collection(collection_name=self._collection)

    async def close(self) -> None:
        """Release the client and, in embedded ``path:`` mode, its on-disk lock,
        so the same store can be reopened sequentially. The build script reads
        the text corpus, encodes, then reopens to upsert, and local mode rejects two
        live clients on one path. No-op for a shared client passed in by the
        serve wiring; the text store owns that one."""
        if self._owns_client:
            await self._client.close()

    async def count(self) -> int:
        """Number of page points; 0 if the collection doesn't exist.

        The serve path uses this to decide whether the visual leg is available
        (empty => degrade to text-only); the build script uses it for idempotent
        re-runs.
        """
        existing = await self._client.get_collections()
        if not any(c.name == self._collection for c in existing.collections):
            return 0
        result = await self._client.count(collection_name=self._collection, exact=True)
        return int(result.count)

    async def upsert_pages(self, pages: list[tuple[str, int, list[list[float]]]]) -> None:
        """Upsert page multivectors. Each item is ``(paper_id, page_no, vectors)``
        where ``vectors`` is the page's ``[n_patches, dim]`` embedding as a
        list-of-rows. The point id is deterministic in the chunk-id so re-builds
        overwrite rather than duplicate."""
        if not pages:
            return
        points = [
            PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_OID,
                        _PAGE_CHUNK_FMT.format(paper_id=paper_id, page_no=page_no),
                    )
                ),
                vector=vectors,
                payload={
                    "chunk_id": _PAGE_CHUNK_FMT.format(paper_id=paper_id, page_no=page_no),
                    "paper_id": paper_id,
                    "page_number": page_no,
                },
            )
            for paper_id, page_no, vectors in pages
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        query_multivector: list[list[float]],
        top_k: int,
        *,
        paper_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Top-``top_k`` pages by MaxSim against the query's multivector.

        ``query_multivector`` is the query's ``[n_q_tokens, dim]`` embedding
        (a 2D vector). Qdrant detects the 2D query and applies the collection's
        MAX_SIM comparator. Results are mapped to the same ``RetrievalResult``
        shape the in-memory leg emits (``source="visual"``, the ``::page``
        chunk-id), so RoutingRetriever fuses them identically.

        ``exact`` makes a server score every page, as embedded mode and
        colpali's ``score_multi_vector`` do, instead of walking an HNSW graph
        that a collection created before ``m=0`` may carry. Approximate search
        moved committed MMDocIR runs (ADR 0032, 2026-09-24 amendment).
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
            query=query_multivector,
            limit=top_k,
            query_filter=qdrant_filter,
            search_params=SearchParams(exact=True),
        )
        results: list[RetrievalResult] = []
        for point in response.points:
            payload = point.payload or {}
            paper_id = str(payload.get("paper_id", ""))
            page_no = int(payload.get("page_number", 0))
            results.append(
                RetrievalResult(
                    chunk_id=str(payload.get("chunk_id", "")),
                    paper_id=paper_id,
                    score=float(point.score),
                    text=f"[Page image {paper_id} p{page_no}]",
                    page_numbers=[page_no],
                    source="visual",
                    score_kind="maxsim",
                )
            )
        return results
