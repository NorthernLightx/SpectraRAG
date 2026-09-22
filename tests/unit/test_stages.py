"""Per-stage retrieval timings reach the eval run and the /query response."""

from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient

from src.api.deps import get_retriever
from src.api.main import create_app
from src.eval.report import render_markdown
from src.eval.runner import evaluate
from src.observability.stages import collect_stages, stage
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, GoldenQuery, GoldenSet, Query, RetrievalResult
from tests.fakes import FakeEmbedder


class _VisualLeg:
    """Times its encode in a worker thread, like VisualRetriever."""

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        with stage("visual_encode"):
            await asyncio.to_thread(time.sleep, 0.02)
        return [
            RetrievalResult(
                chunk_id="p1::p1::page",
                paper_id="p1",
                score=20.0,
                text="",
                page_numbers=[1],
                source="visual",
                score_kind="maxsim",
            )
        ]


async def _router() -> RoutingRetriever:
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="stages", dim=8)
    await store.ensure_collection()
    chunks = [Chunk(chunk_id="p1::p1::c0", paper_id="p1", page_numbers=[1], text="alpha")]
    await store.upsert_chunks(chunks, await embedder.embed_texts(["alpha"]))
    bm25 = Bm25Index()
    bm25.add(chunks)
    text = PipelineRetriever(
        embedder=embedder,
        vectorstore=store,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
        reranker=BgeReranker(scorer=lambda pairs: [0.9] * len(pairs)),
    )
    return RoutingRetriever(text=text, visual=_VisualLeg(), mode="hybrid")


async def test_both_legs_and_their_threads_report_into_one_collector() -> None:
    router = await _router()
    with collect_stages() as stages:
        await router.retrieve(Query(text="alpha", top_k=2))
    assert {"embed", "dense", "bm25", "rerank", "visual_encode", "fuse"} <= set(stages)
    assert stages["visual_encode"] >= 15


async def test_no_collector_means_no_recording() -> None:
    with stage("orphan"):
        pass
    with collect_stages() as stages:
        pass
    assert stages == {}


async def test_eval_run_records_stage_timings() -> None:
    golden = GoldenSet(
        name="g",
        version="v1",
        queries=[GoldenQuery(query_id="q", text="alpha", paper_id="p1", category="factual")],
    )
    run = await evaluate(retriever=await _router(), golden_set=golden, top_k=2)
    [pq] = run.per_query
    assert pq.stage_ms is not None and "rerank" in pq.stage_ms
    assert "| rerank |" in render_markdown(run)


async def test_query_response_carries_stage_timings() -> None:
    app = create_app(log_file=None)
    router = await _router()
    app.dependency_overrides[get_retriever] = lambda: router
    body = TestClient(app).post("/query", json={"text": "alpha", "top_k": 2}).json()
    assert {"embed", "rerank", "visual_encode"} <= set(body["stage_ms"])
