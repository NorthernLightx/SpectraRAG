"""QdrantVisualStore — multivector round-trip + MaxSim ranking on `:memory:`.

Proves the embedded qdrant-client path: a multivector collection accepts page
multivectors and a 2D query, and ranks by the MAX_SIM comparator. This is the
mechanism the persisted visual leg relies on (ADR 0028).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from qdrant_client.http.models import VectorParams

from src.rag.retrievers.visual import VisualRetriever
from src.rag.visual_store import QdrantVisualStore
from src.types import Query


async def _vector_params(store: QdrantVisualStore, collection: str) -> VectorParams:
    params = (await store._client.get_collection(collection)).config.params.vectors
    assert isinstance(params, VectorParams)
    return params


async def test_upsert_and_search_ranks_planted_page() -> None:
    store = QdrantVisualStore(":memory:", "test_visual", dim=4)
    await store.ensure_collection()
    await store.upsert_pages(
        [
            ("paperA", 1, [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            ("paperB", 2, [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        ]
    )
    assert await store.count() == 2

    results = await store.search([[1.0, 0.0, 0.0, 0.0]], top_k=2)
    assert results, "expected at least one hit"
    top = results[0]
    assert top.paper_id == "paperA"
    assert top.chunk_id == "paperA::p1::page"
    assert top.source == "visual"
    assert top.page_numbers == [1]


async def test_paper_filter_restricts_results() -> None:
    store = QdrantVisualStore(":memory:", "test_visual_filter", dim=4)
    await store.ensure_collection()
    await store.upsert_pages(
        [
            ("paperA", 1, [[1.0, 0.0, 0.0, 0.0]]),
            ("paperB", 2, [[1.0, 0.0, 0.0, 0.0]]),
        ]
    )
    results = await store.search([[1.0, 0.0, 0.0, 0.0]], top_k=5, paper_filter="paperB")
    assert results
    assert {r.paper_id for r in results} == {"paperB"}


async def test_creation_needs_the_encoder_dim() -> None:
    """No ColQwen2 (128) default: a store created without the encoder's
    per-token dim refuses to create a collection. Vultron emits 320."""
    with pytest.raises(ValueError, match="dim"):
        await QdrantVisualStore(":memory:", "no_dim").ensure_collection()

    store = QdrantVisualStore(":memory:", "dim_320", dim=320)
    await store.ensure_collection()
    assert (await _vector_params(store, "dim_320")).size == 320
    await store.upsert_pages([("paperA", 1, [[0.1] * 320, [0.2] * 320])])
    hits = await store.search([[1.0] * 320], top_k=1)
    assert hits[0].chunk_id == "paperA::p1::page"


async def test_collection_is_created_on_disk_without_an_hnsw_graph() -> None:
    """A Qdrant server keeps page vectors in RAM unless on_disk is set at
    creation, and cannot change it afterwards. m=0 builds no HNSW graph."""
    store = QdrantVisualStore(":memory:", "on_disk", dim=4)
    await store.ensure_collection()

    params = await _vector_params(store, "on_disk")
    assert params.on_disk is True
    assert params.hnsw_config is not None
    assert params.hnsw_config.m == 0


async def test_search_asks_for_exact_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a server, a collection with an HNSW graph answers approximately unless
    the query asks for exact search; embedded mode is always exact."""
    store = QdrantVisualStore(":memory:", "exact", dim=4)
    await store.ensure_collection()
    seen: dict[str, Any] = {}
    real_query_points = store._client.query_points

    async def _spy(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return await real_query_points(**kwargs)

    monkeypatch.setattr(store._client, "query_points", _spy)
    await store.search([[1.0, 0.0, 0.0, 0.0]], top_k=1)

    assert seen["search_params"].exact is True


class _StubBatch:
    """Mimics a transformers BatchEncoding: `.to(device)` + `**batch` unpacking."""

    def to(self, _device: str) -> _StubBatch:
        return self

    def keys(self) -> list[str]:
        return []

    def __getitem__(self, _key: str) -> Any:
        raise KeyError(_key)


class _StubQueryProcessor:
    def process_queries(self, _queries: list[str]) -> _StubBatch:
        return _StubBatch()


class _FixedQueryModel:
    """Returns a fixed `[1, n_q, dim]` query embedding regardless of input, so
    the store-backed retrieve path is exercised with a known query vector."""

    def __init__(self, vec: list[list[float]]) -> None:
        self._t = torch.tensor([vec], dtype=torch.float32)

    def __call__(self, **_kwargs: Any) -> torch.Tensor:
        return self._t


async def test_store_backed_retriever_ranks_aligned_page() -> None:
    """VisualRetriever with a `store` encodes the query and ranks via Qdrant
    MaxSim — the deploy path. The query aligns with paperA's page vector, so
    paperA ranks first."""
    store = QdrantVisualStore(":memory:", "rt_visual", dim=4)
    await store.ensure_collection()
    await store.upsert_pages(
        [
            ("paperA", 1, [[1.0, 0.0, 0.0, 0.0]]),
            ("paperB", 2, [[0.0, 1.0, 0.0, 0.0]]),
        ]
    )
    retriever = VisualRetriever(
        model=_FixedQueryModel([[1.0, 0.0, 0.0, 0.0]]),
        processor=_StubQueryProcessor(),
        store=store,
        device="cpu",
    )

    out = await retriever.retrieve(Query(text="q", top_k=2))

    assert out
    assert out[0].paper_id == "paperA"
    assert out[0].chunk_id == "paperA::p1::page"
    assert out[0].source == "visual"
    assert out[0].page_numbers == [1]
