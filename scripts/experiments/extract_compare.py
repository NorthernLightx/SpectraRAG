"""Head-to-head extraction-recall comparison across backends on the clean 25-card
structured-object set (Goal 2026-06-01: efficient + near/above SOTA extraction).

Scores each backend's cached extractions with the same recall matcher, on the same
cards, and prints a side-by-side table + the per-card win/loss vs qwen-cloud (the
cloud-VLM accuracy ceiling). Run any time — scores whatever each cache has so far,
so it gives a live partial comparison while a slow local bench is still running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from scripts.experiments.extraction_recall import _gold_tokens, _present, gold_reliability

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_BACKENDS = {
    "qwen-cloud": "data/eval/runs/extract_bench_cache.json",
    "docling": "data/eval/runs/extract_bench_docling_cache.json",
    "mineru-server": "data/eval/runs/extract_bench_mineru_srv_cache.json",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--targets", type=Path, default=Path("data/eval/audit/_struct_target_qids.json"))
    args = ap.parse_args()

    g = {q["query_id"]: q for q in yaml.safe_load(args.golden.read_text(encoding="utf-8"))["queries"]}
    clean = set(json.loads(args.targets.read_text(encoding="utf-8")))

    # Single-char golds ("9","4") match incidentally under the presence matcher, so
    # the RELIABLE recall set excludes them (2026-06-02 review). Compute the reliable
    # qid set once, from the golden, so every backend is scored on the same cards.
    reliable = {qid for qid in clean
                if gold_reliability(str((g[qid].get("expected_facts") or [""])[0])) == "ok"}

    # backend -> {qid: (hit, failed)} over the clean set; recall reported on reliable+extracted.
    hits: dict[str, dict[str, bool]] = {}
    failed: dict[str, set[str]] = {}
    for be, path in _BACKENDS.items():
        p = Path(path)
        if not p.exists():
            continue
        cache = json.loads(p.read_text(encoding="utf-8"))
        h: dict[str, bool] = {}
        fl: set[str] = set()
        for k, ext in cache.items():
            qid = k.split("::", 1)[1]
            if qid not in clean or not isinstance(ext, str):
                continue
            if ext.startswith("__extract_failed__"):
                fl.add(qid)
                continue  # failed extraction: not scorable, excluded (not a 0)
            gold = str((g[qid].get("expected_facts") or [""])[0])
            toks = _gold_tokens(gold)
            h[qid] = bool(toks) and all(_present(t, ext) for t in toks)
        hits[be] = h
        failed[be] = fl

    print(f"EXTRACTION-RECALL HEAD-TO-HEAD ({len(clean)} structured cards, "
          f"{len(reliable)} reliable after dropping single-char golds)\n")
    print(f"  {'backend':16}{'reliable n':>11}{'recall':>9}{'failed':>8}")
    for be, h in hits.items():
        rel = {q: v for q, v in h.items() if q in reliable}
        n = len(rel)
        r = sum(rel.values()) / n if n else 0.0
        print(f"  {be:16}{n:>11}{r:>9.3f}{len(failed.get(be, set())):>8}")

    # head-to-head vs qwen-cloud on the RELIABLE overlap (both extracted, single-char golds dropped)
    if "qwen-cloud" in hits:
        q = hits["qwen-cloud"]
        for be, h in hits.items():
            if be == "qwen-cloud":
                continue
            overlap = [qid for qid in h if qid in q and qid in reliable]
            if not overlap:
                continue
            be_r = sum(h[qid] for qid in overlap) / len(overlap)
            q_r = sum(q[qid] for qid in overlap) / len(overlap)
            won = [qid for qid in overlap if h[qid] and not q[qid]]
            lost = [qid for qid in overlap if q[qid] and not h[qid]]
            print(f"\n  {be} vs qwen-cloud on {len(overlap)} shared cards: "
                  f"{be_r:.3f} vs {q_r:.3f}  (+{len(won)} -{len(lost)})")
            if won:
                print(f"    {be} wins: {[w.split('_')[1] for w in won]}")
            if lost:
                print(f"    qwen wins: {[w.split('_')[1] for w in lost]}")


if __name__ == "__main__":
    main()
