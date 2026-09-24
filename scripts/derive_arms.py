"""Derive every retrieval arm from one run's recorded leg rankings.

An eval run through the router records each leg's ranked chunk ids
(`per_query[].leg_chunk_ids`). From those, this writes one run JSON per arm,
with no model calls and no corpus:

  text-only     the text leg's top-k
  visual-only   the visual leg's top-k
  hybrid-w<W>   page-level weighted RRF of the two legs, once per --weight

Every arm reads the same leg outputs, so the arms are paired per query and a
difference between two of them is the fusion policy alone. Arms taken from
separate eval invocations also differ by run-to-run leg noise (ADR 0032,
2026-09-23 amendment).

Legs recorded at top-k hand the fusion only top-k pages each, and any weight
above ~1.15 then returns the visual leg's pages (ADR 0023, 2026-09-23
amendment). Record deeper legs with `eval_run --fusion-depth 50`.

Run:
  uv run python -m scripts.derive_arms --run data/eval/runs/run-<ts>.json \\
      --golden data/golden/mmdocir-v1.yaml --out-dir data/eval/runs/arms \\
      --weight 1 --weight 2 --weight 5

Runs that predate leg recording can be paired from a text-only and a
visual-only run instead (`--legs-from TEXT VISUAL`), with that leg noise.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.rescore_mmlb_pages import rescore
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.rag.retrievers.routing import fused_page_order


def _page(chunk_id: str) -> str:
    return "::".join(chunk_id.split("::")[:2])


def hybrid_ids(
    text_ids: list[str], visual_ids: list[str], *, top_k: int, visual_weight: float
) -> list[str]:
    """The fused arm's result ids: for each fused page, the text leg's first
    chunk on it (its best, the leg being sorted), else the visual page id.
    Mirrors `RoutingRetriever._fuse_page_level`."""
    first_text: dict[str, str] = {}
    for chunk_id in text_ids:
        first_text.setdefault(_page(chunk_id), chunk_id)
    visual_by_page = {_page(v): v for v in visual_ids}
    return [
        first_text.get(page) or visual_by_page[page]
        for page in fused_page_order(text_ids, visual_ids, top_k=top_k, visual_weight=visual_weight)
    ]


def legs_from_runs(text_run: dict[str, Any], visual_run: dict[str, Any]) -> dict[str, Any]:
    """A run-shaped dict whose leg rankings come from two single-leg runs."""
    visual_by_qid = {pq["query_id"]: pq["retrieved_chunk_ids"] for pq in visual_run["per_query"]}
    per_query = [
        {
            **pq,
            "leg_chunk_ids": {
                "text": pq["retrieved_chunk_ids"],
                "visual": visual_by_qid.get(pq["query_id"], []),
            },
        }
        for pq in text_run["per_query"]
    ]
    run_id = f"{text_run['run_id']}+{visual_run['run_id']}"
    return {**text_run, "run_id": run_id, "per_query": per_query}


def derive(
    run: dict[str, Any], golden: dict[str, Any], *, top_k: int, weights: list[float]
) -> dict[str, dict[str, Any]]:
    """Arm name -> run dict, scored like the committed baselines."""
    legs = [(pq, pq.get("leg_chunk_ids") or {}) for pq in run["per_query"]]
    partial = [pq["query_id"] for pq, leg in legs if not {"text", "visual"} <= set(leg)]
    if partial:
        # A text-routed query (category or cascade mode) or a visual-leg failure
        # records one leg; scoring its missing leg as empty would invent a miss.
        raise SystemExit(
            f"{len(partial)} queries lack a text or visual leg ranking (first: "
            f"{partial[0]}). Record with eval_run --router --force-route hybrid."
        )

    arms: dict[str, Any] = {
        "text-only": lambda leg: leg.get("text", [])[:top_k],
        "visual-only": lambda leg: leg.get("visual", [])[:top_k],
    }
    for w in weights:
        arms[f"hybrid-w{w:g}"] = lambda leg, w=w: hybrid_ids(
            leg.get("text", []), leg.get("visual", []), top_k=top_k, visual_weight=w
        )

    source_config = run.get("config") or {}
    page_labelled = any(q.get("relevant_pages") for q in golden["queries"])
    chunk_labels = {q["query_id"]: q.get("relevant_chunk_ids") or [] for q in golden["queries"]}
    out: dict[str, dict[str, Any]] = {}
    for name, pick in arms.items():
        per_query = []
        for pq, leg in legs:
            ids = pick(leg)
            relevant = chunk_labels.get(pq["query_id"], [])
            per_query.append(
                {
                    "query_id": pq["query_id"],
                    "category": pq["category"],
                    "text": pq["text"],
                    "retrieved_chunk_ids": ids,
                    "retrieval": {
                        "ndcg_at_5": ndcg_at_k(relevant, ids, k=5),
                        "recall_at_10": recall_at_k(relevant, ids, k=10),
                        "mrr": reciprocal_rank(relevant, ids),
                    },
                    # Per-arm latency does not exist: the legs ran once.
                    "latency_ms": 0,
                }
            )
        arm_run = {
            "run_id": f"{run['run_id']}-{name}",
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "golden_set_name": run["golden_set_name"],
            "golden_set_version": run["golden_set_version"],
            "config": {
                "derived_from": run["run_id"],
                "arm": name,
                "top_k": top_k,
                "retrieval_fingerprint": source_config.get("retrieval_fingerprint"),
                "retrieval_config": source_config.get("retrieval_config"),
            },
            "per_query": per_query,
        }
        out[name] = rescore(arm_run, golden) if page_labelled else arm_run
    return out


def _mean(run: dict[str, Any], metric: str) -> float | None:
    values = [
        pq["retrieval"][metric]
        for pq in run["per_query"]
        if pq.get("category") != "out_of_corpus"
        and (pq.get("retrieval") or {}).get(metric) is not None
    ]
    return sum(values) / len(values) if values else None


def read_run(path: Path) -> dict[str, Any]:
    """A run JSON, plain or gzipped. Committed leg recordings are gzipped: at
    fusion depth 50 they exceed the repo's 1 MB cap on added files."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    run: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return run


def run_stem(path: Path) -> str:
    """The file name without `.json` or `.json.gz`."""
    name = path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="A run recorded with leg rankings.")
    source.add_argument(
        "--legs-from",
        type=Path,
        nargs=2,
        metavar=("TEXT_RUN", "VISUAL_RUN"),
        help="Pair a text-only run and a visual-only run by query_id.",
    )
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--weight",
        type=float,
        action="append",
        help="Visual-leg RRF weight for a hybrid arm. Repeat for several. Default 1.",
    )
    args = parser.parse_args()

    if args.run is not None:
        run = read_run(args.run)
        stem = run_stem(args.run)
    else:
        text_path, visual_path = args.legs_from
        run = legs_from_runs(read_run(text_path), read_run(visual_path))
        stem = f"{run_stem(text_path)}+{run_stem(visual_path)}"
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    arms = derive(run, golden, top_k=args.top_k, weights=args.weight or [1.0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'arm':<16}{'recall@10':>10}{'nDCG@5':>9}")
    for name, arm_run in arms.items():
        path = args.out_dir / f"{stem}.{name}.json"
        path.write_text(json.dumps(arm_run, indent=2), encoding="utf-8", newline="\n")
        recall = _mean(arm_run, "recall_at_10")
        ndcg = _mean(arm_run, "ndcg_at_5")
        print(
            f"{name:<16}"
            f"{'n/a' if recall is None else f'{recall:.4f}':>10}"
            f"{'n/a' if ndcg is None else f'{ndcg:.4f}':>9}"
        )
    print(f"Wrote {len(arms)} arms to {args.out_dir}")


if __name__ == "__main__":
    main()
