"""Unit tests for scripts/experiments/visual_rerank_probe.py with a fake scorer.

No model, no GPU: the fake scores a page image by its file stem, so ordering,
tie-breaking, cache resume and the output run schema are checked on their own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.experiments.paired_arm_compare import _scores
from scripts.experiments.visual_rerank_probe import (
    Candidate,
    ScoreCache,
    SlowPairGuard,
    candidate_pages,
    load_candidates,
    merge_runs,
    page_id_of,
    rerank,
    resolve_image,
    run_probe,
    score_all,
    stratified_subset,
)
from src.types.eval import EvalRun


class FakeScorer:
    """Scores an image by its stem (`<paper>_p<N>`) from a table; 0.0 if absent."""

    def __init__(self, table: dict[str, float], *, fail_after: int | None = None) -> None:
        self.table = table
        self.fail_after = fail_after
        self.calls: list[str] = []
        self.fingerprint = "test/fake-scorer@0#0"
        self.config: dict[str, Any] = {"model": "test/fake-scorer"}

    def score(self, query: str, images: Sequence[Path]) -> list[float]:
        if self.fail_after is not None and len(self.calls) + len(images) > self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls.extend(p.stem for p in images)
        return [self.table.get(p.stem, 0.0) for p in images]

    def stats(self) -> dict[str, Any]:
        return {}


def _visual(paper: str, pages: Sequence[int]) -> list[str]:
    return [f"{paper}::p{n}::page" for n in pages]


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """A two-paper, four-query run with a visual leg of five pages each."""
    pages_dir = tmp_path / "pages"
    for paper in ("docA", "docB"):
        (pages_dir / paper).mkdir(parents=True)
        for n in range(1, 8):
            (pages_dir / paper / f"{paper}_p{n}.jpg").write_bytes(b"")
    queries = [
        ("q1", "figure", "docA", [1, 2, 3, 4, 5], [4]),
        ("q2", "table", "docA", [5, 4, 3, 2, 1], [5]),
        ("q3", "factual", "docB", [2, 3, 4, 5, 6], [2, 7]),
        ("q4", "figure", "docB", [6, 5, 4, 3, 2], [3, 5]),
    ]
    run = {
        "run_id": "src",
        "started_at": "2026-09-24T00:00:00",
        "finished_at": "2026-09-24T01:00:00",
        "golden_set_name": "toy",
        "golden_set_version": "v1",
        "config": {"retrieval_fingerprint": "abc123"},
        "per_query": [
            {
                "query_id": qid,
                "category": cat,
                "text": f"question {qid}",
                "retrieved_chunk_ids": [],
                "leg_chunk_ids": {"visual": _visual(paper, ranks), "text": []},
            }
            for qid, cat, paper, ranks, _ in queries
        ],
    }
    golden = {
        "name": "toy",
        "version": "v1",
        "queries": [
            {"query_id": qid, "paper_id": paper, "category": cat, "relevant_pages": gold}
            for qid, cat, paper, _, gold in queries
        ],
    }
    return run, golden, pages_dir


def test_page_id_of_maps_chunks_and_pages() -> None:
    assert page_id_of("docA::p3::c7") == "docA::p3::page"
    assert page_id_of("docA::p03::page") == "docA::p3::page"
    assert page_id_of("no-page-marker") is None


def test_candidate_pages_dedupes_in_rank_order_and_cuts_at_depth() -> None:
    ranked = ["d::p2::c1", "d::p2::c5", "d::p9::page", "junk", "d::p1::c0", "d::p4::c0"]
    assert candidate_pages(ranked, depth=3) == ["d::p2::page", "d::p9::page", "d::p1::page"]


def test_rerank_sorts_by_score_and_breaks_ties_by_first_stage_rank() -> None:
    pages = ["a", "b", "c", "d"]
    scores = {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.5}
    assert rerank(pages, scores) == ["b", "c", "d", "a"]
    assert rerank(pages, dict.fromkeys(pages, 1.0)) == pages


def test_load_candidates_fails_loudly_on_a_missing_leg(
    corpus: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    run, _, _ = corpus
    run["per_query"][2]["leg_chunk_ids"] = {"text": ["docB::p1::c0"]}
    with pytest.raises(SystemExit, match="q3"):
        load_candidates(run, leg="visual", depth=5)


def test_load_candidates_reads_retrieved_ids_for_a_visual_only_run(
    corpus: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    run, _, _ = corpus
    for pq in run["per_query"]:
        pq["retrieved_chunk_ids"] = pq.pop("leg_chunk_ids")["visual"]
    cands = load_candidates(run, leg="retrieved", depth=2)
    assert cands[0].pages == ["docA::p1::page", "docA::p2::page"]


def test_stratified_subset_is_proportional_exact_and_seeded() -> None:
    items = [(f"f{i}", "factual") for i in range(50)]
    items += [(f"g{i}", "figure") for i in range(30)]
    items += [(f"t{i}", "table") for i in range(20)]
    pick = stratified_subset(items, 10, seed=1)
    assert len(pick) == 10
    cats = [qid[0] for qid in pick]
    assert (cats.count("f"), cats.count("g"), cats.count("t")) == (5, 3, 2)
    assert pick == stratified_subset(items, 10, seed=1)
    assert pick != stratified_subset(items, 10, seed=2)
    # Output keeps the input order, so arms list queries as the source run did.
    order = {qid: i for i, (qid, _) in enumerate(items)}
    assert pick == sorted(pick, key=order.__getitem__)
    assert len(stratified_subset(items, 7, seed=1)) == 7


def test_resolve_image_fails_loudly_when_the_render_is_missing(tmp_path: Path) -> None:
    (tmp_path / "docA").mkdir()
    (tmp_path / "docA" / "docA_p2.png").write_bytes(b"")
    assert resolve_image(tmp_path, "docA::p2::page").name == "docA_p2.png"
    with pytest.raises(FileNotFoundError):
        resolve_image(tmp_path, "docA::p3::page")


def test_cache_resume_scores_only_the_missing_pairs(
    corpus: tuple[dict[str, Any], dict[str, Any], Path], tmp_path: Path
) -> None:
    run, _, pages_dir = corpus
    cands = load_candidates(run, leg="visual", depth=5)
    table = {f"docA_p{n}": float(n) for n in range(1, 8)} | {f"docB_p{n}": -n for n in range(8)}
    cache_path = tmp_path / "scores.jsonl"

    # q1 scores in batches of 2, 2, 1; q2's first batch lands (7 pairs), its
    # second batch crashes.
    crashing = FakeScorer(table, fail_after=7)
    with pytest.raises(RuntimeError, match="simulated crash"):
        score_all(
            cands, crashing, ScoreCache(cache_path, crashing.fingerprint), pages_dir, batch_size=2
        )
    assert len(crashing.calls) == 7

    resumed = FakeScorer(table)
    cache = ScoreCache(cache_path, resumed.fingerprint)
    assert len(cache) == 7
    assert score_all(cands, resumed, cache, pages_dir, batch_size=2) == 13
    # Resumes at q2's crashed batch (its leg runs p5, p4, p3, p2, p1).
    assert resumed.calls[:3] == ["docA_p3", "docA_p2", "docA_p1"]

    # Nothing left to do on a third pass, and the scores equal a clean run's.
    again = FakeScorer(table)
    done = score_all(
        cands, again, ScoreCache(cache_path, again.fingerprint), pages_dir, batch_size=2
    )
    assert done == 0
    clean_path = tmp_path / "clean.jsonl"
    clean = ScoreCache(clean_path, again.fingerprint)
    score_all(cands, FakeScorer(table), clean, pages_dir, batch_size=5)
    reloaded = ScoreCache(cache_path, again.fingerprint)
    for c in cands:
        for p in c.pages:
            assert reloaded.score(c.query_id, p) == clean.score(c.query_id, p)


def test_cache_cuts_a_torn_last_line_and_keeps_appending_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    cache = ScoreCache(path, "fp")
    cache.add("q1", ["a::p1::page"], [1.5], ms=10.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"query_id": "q1", "page_id": "a::p2::pa')
    cache = ScoreCache(path, "fp")
    assert len(cache) == 1
    cache.add("q1", ["a::p2::page"], [0.5], ms=10.0)
    assert len(ScoreCache(path, "fp")) == 2


def test_cache_keeps_one_append_handle_until_closed(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    cache = ScoreCache(path, "fp")
    cache.add("q1", ["a::p1::page"], [1.0], ms=1.0)
    handle = cache._fh
    cache.add("q1", ["a::p2::page"], [2.0], ms=1.0)
    assert cache._fh is handle
    # Each batch is on disk before the next one starts, handle still open.
    assert len(ScoreCache(path, "fp")) == 2
    cache.close()
    assert handle is not None and handle.closed
    cache.add("q1", ["a::p3::page"], [3.0], ms=1.0)
    cache.close()
    assert len(ScoreCache(path, "fp")) == 3


def test_cache_refuses_scores_from_another_scorer(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    ScoreCache(path, "model-a@rev#1").add("q1", ["a::p1::page"], [1.0], ms=1.0)
    with pytest.raises(SystemExit, match="model-a"):
        ScoreCache(path, "model-a@rev#2")


def test_slow_pair_guard_stops_a_run_that_is_spilling() -> None:
    guard = SlowPairGuard(limit_s=3.0, warmup=2, window=10)
    guard.observe(30.0, 2)  # warmup is ignored
    for _ in range(12):
        guard.observe(0.4, 1)
    with pytest.raises(SystemExit, match="s/pair"):
        for _ in range(12):
            guard.observe(5.0, 1)


def _write_arms(
    corpus: tuple[dict[str, Any], dict[str, Any], Path],
    tmp_path: Path,
    scorer: FakeScorer,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run, golden, pages_dir = corpus
    first_path, reranked_path = run_probe(
        run=run,
        run_stem="toy",
        golden=golden,
        scorer=scorer,
        pages_dir=pages_dir,
        out_dir=tmp_path / "out",
        depth=5,
        leg="visual",
        seed=3,
        batch_size=2,
        **{"subset": None, **kwargs},
    )
    first: dict[str, Any] = json.loads(first_path.read_text(encoding="utf-8"))
    reranked: dict[str, Any] = json.loads(reranked_path.read_text(encoding="utf-8"))
    return first, reranked


def test_output_arms_are_eval_runs_paired_over_the_same_candidates(
    corpus: tuple[dict[str, Any], dict[str, Any], Path], tmp_path: Path
) -> None:
    table = {"docA_p4": 2.0, "docA_p5": 1.0, "docB_p5": 3.0, "docB_p3": 3.0}
    first, reranked = _write_arms(corpus, tmp_path, FakeScorer(table))

    for arm in (first, reranked):
        EvalRun.model_validate(arm)
        assert arm["config"]["derived_from"] == "src"
        assert arm["config"]["retrieval_fingerprint"] == "abc123"
    assert [q["query_id"] for q in first["per_query"]] == ["q1", "q2", "q3", "q4"]
    assert [q["query_id"] for q in reranked["per_query"]] == ["q1", "q2", "q3", "q4"]
    for a, b in zip(first["per_query"], reranked["per_query"], strict=True):
        assert sorted(a["retrieved_chunk_ids"]) == sorted(b["retrieved_chunk_ids"])
        assert (
            a["retrieval"]["recall_at_5"] == b["retrieval"]["recall_at_5"]
        )  # depth 5: the ceiling

    by_q = {q["query_id"]: q for q in reranked["per_query"]}
    assert by_q["q1"]["retrieved_chunk_ids"][:2] == ["docA::p4::page", "docA::p5::page"]
    assert by_q["q1"]["retrieval"]["mrr"] == 1.0
    assert first["per_query"][0]["retrieval"]["mrr"] == 0.25
    # Tie between docB p5 and p3 keeps the first-stage order (p5 before p3).
    assert by_q["q4"]["retrieved_chunk_ids"][:2] == ["docB::p5::page", "docB::p3::page"]
    # q3 has gold pages 2 and 7; 7 is never retrieved, 2 lands at rank 3.
    q3 = by_q["q3"]["retrieval"]
    assert q3["recall_at_5"] == 0.5
    # rescore takes the ideal from the one retrieved gold page; the standard
    # nDCG the committed baselines hold counts both gold pages.
    assert q3["ndcg_at_5"] == pytest.approx(0.5)
    assert q3["ndcg_at_5_standard"] == pytest.approx(0.5 / (1 + 1 / math.log2(3)))

    for path_run in (first, reranked):
        path = tmp_path / f"{path_run['config']['arm']}.json"
        path.write_text(json.dumps(path_run), encoding="utf-8")
        assert set(_scores(path, "recall_at_5")) == {"q1", "q2", "q3", "q4"}


def test_an_oracle_scorer_reaches_the_ceiling_and_a_reversed_one_falls_below(
    corpus: tuple[dict[str, Any], dict[str, Any], Path], tmp_path: Path
) -> None:
    oracle = {"docA_p4": 1.0, "docA_p5": 1.0, "docB_p2": 1.0, "docB_p3": 1.0, "docB_p5": 1.0}
    _, best = _write_arms(corpus, tmp_path / "o", FakeScorer(oracle))
    for pq in best["per_query"]:
        hit = pq["retrieval"]["recall_at_5"] > 0
        assert pq["retrieval"]["mrr"] == (1.0 if hit else 0.0)

    reverse = {f"{d}_p{n}": float(-n) for d in ("docA",) for n in range(8)}
    reverse |= {f"docB_p{n}": float(-10 + n) for n in range(8)}
    _, worse = _write_arms(corpus, tmp_path / "r", FakeScorer(reverse))
    mrr = {q["query_id"]: q["retrieval"]["mrr"] for q in worse["per_query"]}
    assert mrr["q2"] < 1.0  # gold p5 was first; ascending page order buries it


def test_subset_run_scores_only_the_subset(
    corpus: tuple[dict[str, Any], dict[str, Any], Path], tmp_path: Path
) -> None:
    scorer = FakeScorer({})
    first, reranked = _write_arms(corpus, tmp_path, scorer, subset=2)
    assert len(first["per_query"]) == len(reranked["per_query"]) == 2
    assert len(scorer.calls) == 10
    assert reranked["config"]["subset"] == {"n": 2, "seed": 3}


def test_candidate_is_frozen() -> None:
    cand = Candidate("q", "figure", "t", ["a::p1::page"])
    with pytest.raises(AttributeError):
        cand.query_id = "other"  # type: ignore[misc]


def test_merge_runs_joins_slices_and_refuses_mixed_stacks(
    corpus: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    run, _, _ = corpus
    head = {**run, "run_id": "s0", "per_query": run["per_query"][:2]}
    tail = {**run, "run_id": "s1", "per_query": run["per_query"][2:]}
    tail["finished_at"] = "2026-09-24T03:00:00"
    merged = merge_runs([head, tail])
    assert [q["query_id"] for q in merged["per_query"]] == ["q1", "q2", "q3", "q4"]
    assert merged["run_id"] == "s0+s1"
    assert merged["finished_at"] == "2026-09-24T03:00:00"
    assert merge_runs([run]) is run

    other_stack = {**tail, "config": {"retrieval_fingerprint": "different"}}
    with pytest.raises(SystemExit, match="retrieval_fingerprint"):
        merge_runs([head, other_stack])
    with pytest.raises(SystemExit, match="q1"):
        merge_runs([head, head])
    with pytest.raises(SystemExit, match="golden_set_version"):
        merge_runs([head, {**tail, "golden_set_version": "v2"}])
