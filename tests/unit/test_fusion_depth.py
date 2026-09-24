"""Fusion depth, recorded leg rankings, and arms derived from them."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.derive_arms import derive, hybrid_ids, read_run, run_stem
from src.eval.runner import evaluate
from src.rag.retrievers.routing import (
    RoutingRetriever,
    fused_page_order,
    get_last_leg_ids,
)
from src.types import GoldenQuery, GoldenSet, Query, RetrievalResult


def _text(paper: str, page: int, chunk: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper}::p{page}::c{chunk}",
        paper_id=paper,
        score=1.0 - chunk / 100,
        text="t",
        page_numbers=[page],
        source="pipeline",
        score_kind="rerank",
    )


def _visual(paper: str, page: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper}::p{page}::page",
        paper_id=paper,
        score=20.0 - page / 10,
        text="v",
        page_numbers=[page],
        source="visual",
        score_kind="maxsim",
    )


class _Leg:
    """Returns its first `query.top_k` results and records the depth asked for."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.asked: list[int] = []

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.asked.append(query.top_k)
        return self.results[: query.top_k]


_TEXT = [_text("d", p, i) for i, p in enumerate(range(1, 31))]
_VISUAL = [_visual("d", p) for p in range(30, 0, -1)]


def test_weight_above_the_cliff_returns_the_visual_page_set() -> None:
    """Both legs cut to 10: RRF ranks span 1/61..1/70, so w=1.5 already lets
    every visual page outscore every text-only page."""
    text_ids = [r.chunk_id for r in _TEXT[:10]]
    visual_ids = [r.chunk_id for r in _VISUAL[:10]]
    fused = fused_page_order(text_ids, visual_ids, top_k=10, visual_weight=1.5)
    assert set(fused) == {v.rsplit("::", 1)[0] for v in visual_ids}


async def test_fusion_depth_deepens_the_legs_and_trims_the_output() -> None:
    text, visual = _Leg(_TEXT), _Leg(_VISUAL)
    router = RoutingRetriever(text=text, visual=visual, mode="hybrid", fusion_depth=25)
    results = await router.retrieve(Query(text="q", top_k=5))
    assert text.asked == [25]
    assert visual.asked == [25]
    assert len(results) == 5
    legs = get_last_leg_ids()
    assert legs is not None
    assert len(legs["text"]) == 25
    assert len(legs["visual"]) == 25


async def test_without_fusion_depth_the_legs_run_at_top_k() -> None:
    text, visual = _Leg(_TEXT), _Leg(_VISUAL)
    router = RoutingRetriever(text=text, visual=visual, mode="hybrid")
    await router.retrieve(Query(text="q", top_k=5))
    assert text.asked == [5]
    assert visual.asked == [5]


@pytest.mark.parametrize("weight", [1.0, 1.1, 3.0])
async def test_derived_hybrid_arm_matches_the_live_router(weight: float) -> None:
    router = RoutingRetriever(
        text=_Leg(_TEXT), visual=_Leg(_VISUAL), mode="hybrid", visual_fusion_weight=weight
    )
    live = await router.retrieve(Query(text="q", top_k=10))
    derived = hybrid_ids(
        [r.chunk_id for r in _TEXT[:10]],
        [r.chunk_id for r in _VISUAL[:10]],
        top_k=10,
        visual_weight=weight,
    )
    assert [r.chunk_id for r in live] == derived


async def test_runner_records_leg_rankings_and_arms_derive_from_them() -> None:
    router = RoutingRetriever(text=_Leg(_TEXT), visual=_Leg(_VISUAL), mode="hybrid")
    golden = GoldenSet(
        name="g",
        version="v1",
        queries=[
            GoldenQuery(
                query_id="q1", text="q", paper_id="d", category="figure", relevant_pages=[3]
            )
        ],
    )
    run = await evaluate(retriever=router, golden_set=golden, top_k=10)
    [pq] = run.per_query
    assert pq.leg_chunk_ids is not None
    assert set(pq.leg_chunk_ids) == {"text", "visual"}

    arms = derive(
        run.model_dump(mode="json"),
        golden.model_dump(mode="json"),
        top_k=10,
        weights=[1.0],
    )
    assert set(arms) == {"text-only", "visual-only", "hybrid-w1"}
    # Page 3 is rank 3 in the text leg and rank 28 in the visual leg.
    assert arms["text-only"]["per_query"][0]["retrieval"]["recall_at_10"] == 1.0
    assert arms["visual-only"]["per_query"][0]["retrieval"]["recall_at_10"] == 0.0
    hybrid_ids_recorded = arms["hybrid-w1"]["per_query"][0]["retrieved_chunk_ids"]
    assert hybrid_ids_recorded == [
        r.chunk_id for r in await router.retrieve(Query(text="q", top_k=10))
    ]


def test_read_run_accepts_a_gzipped_receipt(tmp_path: Path) -> None:
    """Committed leg recordings are gzipped to stay under the repo's size cap."""
    run = {"run_id": "r", "per_query": [{"query_id": "q1", "leg_chunk_ids": {"text": []}}]}
    plain = tmp_path / "run.json"
    plain.write_text(json.dumps(run), encoding="utf-8")
    packed = tmp_path / "run.json.gz"
    packed.write_bytes(gzip.compress(json.dumps(run).encode("utf-8")))

    assert read_run(plain) == read_run(packed) == run
    assert run_stem(plain) == run_stem(packed) == "run"


def test_derive_refuses_queries_missing_a_leg() -> None:
    """A text-routed query records no visual leg; scoring it as an empty
    visual ranking would invent a miss."""
    run = {
        "run_id": "r",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:00Z",
        "golden_set_name": "g",
        "golden_set_version": "v1",
        "config": {},
        "per_query": [
            {
                "query_id": "q1",
                "category": "figure",
                "text": "q",
                "leg_chunk_ids": {"text": ["d::p1::c0"], "visual": ["d::p1::page"]},
            },
            {
                "query_id": "q2",
                "category": "factual",
                "text": "q",
                "leg_chunk_ids": {"text": ["d::p2::c0"]},
            },
        ],
    }
    golden = {
        "queries": [{"query_id": q, "paper_id": "d", "relevant_pages": [1]} for q in ("q1", "q2")]
    }
    with pytest.raises(SystemExit, match="1 queries lack"):
        derive(run, golden, top_k=10, weights=[1.0])
