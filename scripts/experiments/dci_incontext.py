"""In-context distillation: prime a small fast model with a strong model's traces.

The paper's cost/latency proposal is to distill the agentic think-retrieve loop
into a small open-weight model. Full fine-tuning is out of budget/scope; this is
the cheap, runnable version: capture a few worked read+grep traces from the strong
model (qwen3-235b), inject them as few-shot exemplars, and run the free local
gemma3:4b with them. If the student lifts above its ~28 baseline, the strong
model's strategy transfers in-context — a fast/cheap path toward the big-model band.

Held out: the teacher queries are excluded from the student eval set, so the
exemplars can't leak answers.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.dci_incontext \
        --teacher 4 10 18 --n-eval 25 --out data/eval/runs/dci_incontext.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import src  # noqa: F401 -- loads .env
from scripts.experiments.dci_eval import _load_corpus, _load_queries, _ndcg_at_k
from src.dci.agent import DciAgent, DciResult
from src.dci.tools import CorpusTools
from src.llm.ollama_chat import OllamaChatClient
from src.llm.openrouter import OpenRouterClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_TEACHER_MODEL = "qwen/qwen3-235b-a22b-2507"
_STUDENT_MODEL = "gemma3:4b"


def _format_exemplar(question: str, res: DciResult) -> str:
    """One worked example: the action sequence the strong model used, ending in
    RANK. Observations are dropped — the strategy (what to search, when to rank)
    is what transfers, and full tool output would bloat every student prompt."""
    lines = [f'Worked example. Question: "{question[:180]}"']
    for s in res.steps:
        if s.action in {"SEARCH", "FILTER", "COUNT", "GREP", "READ"}:
            lines.append(f"ACTION: {s.action} {s.arg[:70]}")
    lines.append(f"ACTION: RANK {', '.join(res.ranked_doc_ids[:8])}")
    lines.append("(found the relevant documents)")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    corpus = _load_corpus(args.corpus_dir)
    queries = {q["qid"]: q for q in _load_queries(args.corpus_dir)}
    tools = CorpusTools(corpus)
    key = os.environ["RAG_OPENROUTER_API_KEY"]

    # ---- Teacher: capture strong-model read+grep traces as exemplars ----
    teacher = OpenRouterClient(api_key=key)
    exemplars: list[str] = []
    used: set[str] = set()
    for qid in args.teacher:
        q = queries.get(qid)
        if q is None:
            continue
        agent = DciAgent(tools, teacher, _TEACHER_MODEL, toolset="readgrep", max_steps=12, search_k=10)
        res = await agent.run(q["query"], mode="retrieval", top_k=10)
        nd = _ndcg_at_k(res.ranked_doc_ids, set(q["gold_ids"]), set(q["excluded_ids"]))
        used.add(qid)
        print(f"  teacher {qid}: nDCG={nd:.3f} steps={len(res.steps)} {'(kept)' if nd >= 0.4 else '(skip, weak)'}")
        if nd >= 0.4:
            exemplars.append(_format_exemplar(q["query"], res))
    if not exemplars:
        print("No usable teacher traces — aborting.")
        return 1
    block = "Here are worked examples of the strategy on similar questions:\n\n" + "\n\n".join(exemplars) + "\n\n"

    # ---- Student A/B: gemma3:4b read+grep, same current agent, exemplars on/off ----
    student = OllamaChatClient(base_url=args.ollama)
    eval_qids = [qid for qid in queries if qid not in used][: args.n_eval]

    async def student_score(qid: str, primed: bool) -> float:
        q = queries[qid]
        agent = DciAgent(tools, student, _STUDENT_MODEL, toolset="readgrep",
                         max_steps=16, search_k=10, exemplars=block if primed else "")
        res = await agent.run(q["query"], mode="retrieval", top_k=10)
        return _ndcg_at_k(res.ranked_doc_ids, set(q["gold_ids"]), set(q["excluded_ids"]))

    plain, primed = [], []
    rows: list[dict[str, Any]] = []
    for i, qid in enumerate(eval_qids, 1):
        nd0 = await student_score(qid, primed=False)
        nd1 = await student_score(qid, primed=True)
        plain.append(nd0)
        primed.append(nd1)
        rows.append({"qid": qid, "plain": nd0, "primed": nd1})
        print(f"[{i}/{len(eval_qids)}] {qid}: plain={nd0:.3f} primed={nd1:.3f}")

    b = 100 * sum(plain) / len(plain) if plain else 0.0
    p = 100 * sum(primed) / len(primed) if primed else 0.0
    print(f"\nIn-context distillation ({_STUDENT_MODEL} read+grep, n={len(rows)}, "
          f"{len(exemplars)} {_TEACHER_MODEL} exemplars):")
    print(f"  gemma3, no exemplars : {b:.1f}")
    print(f"  gemma3 + exemplars   : {p:.1f}")
    print(f"  delta (in-context)   : {p - b:+.1f}")
    args.out.write_text(json.dumps({"exemplars": exemplars, "rows": rows,
        "primed_ndcg": p, "plain_ndcg": b}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/dci/bright_biology"))
    ap.add_argument("--teacher", nargs="+", default=["4", "10", "18"], help="qids for teacher exemplars")
    ap.add_argument("--n-eval", type=int, default=25)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/dci_incontext.json"))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
