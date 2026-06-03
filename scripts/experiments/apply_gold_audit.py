"""Bet 1 apply step: consume HUMAN gold-audit verdicts, quantify the ruler noise.

Pairs with build_gold_audit.py. That builder rendered the gold-present failures for
a human to judge; this reads the human's verdicts.json and does three things, all
mechanical — it authors NO ground truth, it only transcribes the human's verdicts
and recomputes arithmetic:

  1. LABEL-NOISE REPORT: how the human classified the failures
     (gold_correct / format_only_mismatch / gold_wrong / gold_unprovable), i.e. how
     much of the measured ceiling was scorer/label noise vs genuine model miss.

  2. CLEAN RE-SCORE: re-score a committed oracle run, crediting the queries the human
     marked format_only_mismatch (model was right, scorer mis-marked) and dropping
     gold_unprovable (unanswerable from the fed page) from the denominator. gold_wrong
     with a human-supplied correction is re-scored against the corrected value via the
     official scorer. Reports answerable ACC before -> after with the delta.

  3. CORRECTED GOLD SLICE: writes the human corrections into a forked golden YAML
     (data/golden/<set>-clean.yaml) so future runs score on the clean ruler. Only
     gold_wrong corrections change a value; everything else is copied verbatim.

Verdict semantics (set by the human in the builder UI):
  - gold_correct        : leave as-is; the original 0 score stands (a real model miss).
  - format_only_mismatch: the model's content was right; credit it (score -> 1.0).
  - gold_wrong          : the gold value was wrong; re-score pred against the human's
                          corrected value, and write the correction into the clean slice.
  - gold_unprovable     : the page can't support gold; drop from the answerable denominator.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.apply_gold_audit \
        --verdicts verdicts.json \
        --scored-run data/eval/runs/free_oracle_scored.json \
        --golden data/golden/mmlongbench-v1.yaml \
        --clean-out data/golden/mmlongbench-clean-v1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.experiments.score_mmlb_qa import _eval_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_VALID = {"gold_correct", "format_only_mismatch", "gold_wrong", "gold_unprovable"}


def _load_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for qid, v in raw.items():
        verdict = v.get("verdict")
        if verdict not in _VALID:
            print(f"  WARN: {qid} has invalid/empty verdict {verdict!r}; skipping")
            continue
        if verdict == "gold_wrong" and not str(v.get("corrected", "")).strip():
            print(f"  WARN: {qid} is gold_wrong but has no corrected value; treating as unprovable")
            v = {**v, "verdict": "gold_unprovable"}
        out[qid] = v
    return out


def _report_noise(verdicts: dict[str, dict[str, Any]]) -> None:
    from collections import Counter

    counts = Counter(v["verdict"] for v in verdicts.values())
    n = len(verdicts)
    print(f"\n=== LABEL-NOISE REPORT (human-adjudicated, n={n}) ===")
    for k in ("gold_correct", "format_only_mismatch", "gold_wrong", "gold_unprovable"):
        c = counts.get(k, 0)
        print(f"  {k:22} {c:3}  ({c / n * 100:.0f}%)" if n else f"  {k:22} {c:3}")
    noise = counts.get("format_only_mismatch", 0) + counts.get("gold_wrong", 0) + counts.get("gold_unprovable", 0)
    print(f"  {'-' * 40}")
    print(f"  ruler noise (not a real model miss): {noise}/{n} ({noise / n * 100:.0f}%)" if n else "")
    print("  (format_only = scorer artifact; gold_wrong/unprovable = bad label)")


def _rescore(
    verdicts: dict[str, dict[str, Any]], scored_run: Path
) -> None:
    rows = json.loads(scored_run.read_text(encoding="utf-8"))
    by_qid = {r["query_id"]: r for r in rows}

    # Answerable subset = the official scorer's positive class (gold != "Not answerable").
    answerable = [r for r in rows if r.get("answer") != "Not answerable"]
    before = sum(r["score"] for r in answerable) / len(answerable) if answerable else 0.0

    # Apply human verdicts to a COPY of the scores.
    adjusted: dict[str, float] = {r["query_id"]: r["score"] for r in answerable}
    dropped: set[str] = set()
    changed = 0
    for qid, v in verdicts.items():
        if qid not in adjusted:
            continue  # verdict for a query not in this run/subset
        verdict = v["verdict"]
        if verdict == "format_only_mismatch":
            if adjusted[qid] < 1.0:
                adjusted[qid] = 1.0
                changed += 1
        elif verdict == "gold_unprovable":
            dropped.add(qid)
            changed += 1
        elif verdict == "gold_wrong":
            row = by_qid[qid]
            new_score = _eval_score(str(v["corrected"]), str(row.get("pred", "")), str(row.get("format", "Str")))
            if new_score != adjusted[qid]:
                adjusted[qid] = new_score
                changed += 1
        # gold_correct: leave the original score.

    kept = [(q, s) for q, s in adjusted.items() if q not in dropped]
    after = sum(s for _, s in kept) / len(kept) if kept else 0.0
    print(f"\n=== CLEAN RE-SCORE ({scored_run.name}, answerable subset) ===")
    print(f"  before (committed gold): ACC {before:.4f}  (n={len(answerable)})")
    print(f"  after  (human-clean)   : ACC {after:.4f}  (n={len(kept)}, dropped {len(dropped)} unprovable)")
    print(f"  delta: {after - before:+.4f}   ({changed} queries adjusted by human verdicts)")
    print("  NOTE: this is the ruler correction, not a model change — same model output, cleaner gold.")


def _write_clean_golden(
    verdicts: dict[str, dict[str, Any]], golden: Path, clean_out: Path
) -> None:
    data = yaml.safe_load(golden.read_text(encoding="utf-8"))
    corrected = 0
    dropped_unprovable = 0
    for q in data.get("queries", []):
        qid = q.get("query_id")
        v = verdicts.get(qid)
        if v is None:
            continue
        if v["verdict"] == "gold_wrong":
            # Transcribe the HUMAN's corrected value; do not invent.
            q["expected_facts"] = [str(v["corrected"]).strip()]
            note = q.get("note", "") or ""
            q["note"] = f"{note} | gold_audit=corrected_by_human"
            corrected += 1
        elif v["verdict"] == "gold_unprovable":
            note = q.get("note", "") or ""
            q["note"] = f"{note} | gold_audit=unprovable_from_page"
            dropped_unprovable += 1
        elif v["verdict"] == "format_only_mismatch":
            note = q.get("note", "") or ""
            q["note"] = f"{note} | gold_audit=format_only_scorer_artifact"
    data["version"] = data.get("version", "v1") + "-clean"
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\n=== CLEAN GOLD SLICE -> {clean_out} ===")
    print(f"  {corrected} gold values corrected, {dropped_unprovable} marked unprovable (human verdicts).")
    print("  Every change traces to a human verdict; the machine transcribed, did not author.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, required=True, help="verdicts.json exported from the audit HTML")
    ap.add_argument("--scored-run", type=Path, default=Path("data/eval/runs/free_oracle_scored.json"))
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--clean-out", type=Path, default=Path("data/golden/mmlongbench-clean-v1.yaml"))
    ap.add_argument("--no-write-golden", action="store_true", help="report only; don't write the clean slice")
    args = ap.parse_args()

    if not args.verdicts.exists():
        print(f"verdicts file not found: {args.verdicts}")
        print("Build the audit HTML, adjudicate in a browser, download verdicts.json first.")
        raise SystemExit(2)

    verdicts = _load_verdicts(args.verdicts)
    if not verdicts:
        print("No valid verdicts found.")
        raise SystemExit(2)

    _report_noise(verdicts)
    if args.scored_run.exists():
        _rescore(verdicts, args.scored_run)
    else:
        print(f"\n(skipping re-score: {args.scored_run} not found)")
    if not args.no_write_golden:
        _write_clean_golden(verdicts, args.golden, args.clean_out)


if __name__ == "__main__":
    main()
