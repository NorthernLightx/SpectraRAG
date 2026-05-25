"""Reconstruct a score_mmlb_qa run JSON from a run_mmlb_qa answer cache.

run_mmlb_qa.py writes the full run JSON only at the END of a complete pass, but
it checkpoints every answer into a resumable --cache keyed
`<model>::k<top_k>::<query_id>`. When a generation pass is PARTIAL (the
qwen3-vl:235b-cloud frontier run is throttled to ~33 calls/day by Ollama's cloud
quota), the run JSON for the answers produced so far doesn't exist yet — only the
cache does. This rebuilds a scorable run JSON from whatever answers are in the
cache, so the partial frontier set can be scored with the official protocol
without waiting for all 149.

It does NOT generate anything (no model call, no retrieval) — it only reshapes
cached answers + golden question text into score_mmlb_qa's `per_query[i].vision.
answer` contract. The MACHINE NEVER AUTHORS GROUND TRUTH: question text comes
from the human golden, the answer is the model's own cached output.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.rebuild_mmlb_run_from_cache \\
        --cache data/eval/runs/mmlb_gen_cloud_w5_cache.SNAPSHOT.json \\
        --golden data/golden/mmlongbench-v1.yaml \\
        --model qwen3-vl:235b-cloud --top-k 5 \\
        --out data/eval/runs/exp_mmlb_gen_frontier_partial.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _gold_answer(query: dict[str, Any]) -> str:
    facts = query.get("expected_facts") or [""]
    return str(facts[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True, help="run_mmlb_qa answer cache JSON")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="run JSON to write")
    parser.add_argument(
        "--model",
        required=True,
        help="model name that prefixes the cache keys (e.g. qwen3-vl:235b-cloud)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="top_k that appears in the cache keys")
    args = parser.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold_by_qid: dict[str, dict[str, Any]] = {q["query_id"]: q for q in golden["queries"]}

    cache: dict[str, dict[str, Any]] = json.loads(args.cache.read_text(encoding="utf-8"))
    prefix = f"{args.model}::k{args.top_k}::"

    records: list[dict[str, Any]] = []
    skipped_no_gold = 0
    for key, vision in cache.items():
        if not key.startswith(prefix):
            continue
        qid = key[len(prefix) :]
        gold = gold_by_qid.get(qid)
        if gold is None:
            skipped_no_gold += 1
            continue
        records.append(
            {
                "query_id": qid,
                "paper_id": gold.get("paper_id"),
                "category": gold.get("category"),
                "query": gold.get("text") or "",
                "gold": _gold_answer(gold),
                "vision": vision,
            }
        )

    out = {
        "config": {
            "reconstructed_from_cache": str(args.cache),
            "vision_model": args.model,
            "top_k": args.top_k,
            "golden": str(args.golden),
            "note": "partial run rebuilt from answer cache; see rebuild_mmlb_run_from_cache.py",
        },
        "per_query": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  reconstructed={len(records)} records  skipped(no gold)={skipped_no_gold}")
    print(
        "\nScore with:\n"
        f"  .venv/Scripts/python.exe -m scripts.experiments.score_mmlb_qa "
        f"--run {args.out} --golden {args.golden} --answer-field vision.answer"
    )


if __name__ == "__main__":
    main()
