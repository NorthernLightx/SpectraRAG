"""Materialise a BRIGHT domain into a DCI corpus + query set.

BRIGHT (xlangai/BRIGHT) is the reasoning-intensive IR benchmark the DCI paper
uses, where dense retrievers fail and lexical+reasoning wins — the cleanest fit
for agentic corpus interaction. Metric: nDCG@10 over `gold_ids`, with
`excluded_ids` removed from the ranking before scoring.

Doc ids in BRIGHT contain slashes and `.txt`; models must echo ids back in RANK,
so we assign short surrogate ids (`d0`, `d1`, …) and keep the real↔surrogate map.
Writes three files under data/dci/bright_<domain>/:
  corpus.jsonl   {id, content}            (surrogate id)
  queries.jsonl  {qid, query, gold_ids, excluded_ids}   (surrogate ids)
  idmap.json     {surrogate: real}        (provenance)

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.dci_prepare_bright --domain biology
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="biology", help="BRIGHT domain (biology, earth_science, economics, robotics)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"data/dci/bright_{args.domain}")
    out.mkdir(parents=True, exist_ok=True)

    docs = load_dataset("xlangai/BRIGHT", "documents", split=args.domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=args.domain)

    real_to_sur: dict[str, str] = {}
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            sur = f"d{i}"
            real_to_sur[d["id"]] = sur
            f.write(json.dumps({"id": sur, "content": d["content"]}) + "\n")

    n_q = 0
    miss_gold = 0
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for e in examples:
            gold = [real_to_sur[g] for g in e["gold_ids"] if g in real_to_sur]
            miss_gold += len(e["gold_ids"]) - len(gold)
            if not gold:
                continue  # unscorable without an in-corpus gold doc
            excluded = [real_to_sur[x] for x in e.get("excluded_ids", []) if x in real_to_sur]
            f.write(json.dumps({
                "qid": str(e["id"]), "query": e["query"],
                "gold_ids": gold, "excluded_ids": excluded,
            }) + "\n")
            n_q += 1

    (out / "idmap.json").write_text(
        json.dumps({s: r for r, s in real_to_sur.items()}), encoding="utf-8"
    )
    print(f"domain={args.domain}")
    print(f"  corpus : {len(real_to_sur)} docs -> {out / 'corpus.jsonl'}")
    print(f"  queries: {n_q} scorable -> {out / 'queries.jsonl'}"
          + (f"  ({miss_gold} gold ids not in corpus, dropped)" if miss_gold else ""))


if __name__ == "__main__":
    main()
