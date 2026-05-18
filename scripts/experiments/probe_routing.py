"""Routing headroom probe: is a better query classifier worth a study?

The router sends figure/table/multi_hop queries to the visual leg and the
rest to the text leg. Phase 1 of the reranker DoE empirically confirmed the
text leg scores ~0.0 on every MMLongBench figure/table query, so a
figure/table query misrouted to text is a near-certain miss. Therefore the
classifier's misroute rate on those queries is a direct lower bound on the
accuracy a perfect router would recover -- computable with no GPU, no
ColQwen2, no retrieval: just the golden labels (the oracle) vs the
classifier's decision.

Compares regex `classify_query` and (optionally, cheap) the Ollama
LLMQueryClassifier against the oracle. This re-checks the repo's own
"~75% regex miss rate" docstring claim directly rather than trusting it
(the same skepticism that caught the false 5.5 s reranker number).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from src.eval.golden_set import load_golden_set
from src.rag.retrievers.routing import classify_query

HYBRID = {"figure", "table", "multi_hop"}  # -> visual leg (calibrate_cascade)
GOLDENS = [("v3", "data/golden/v3.yaml"), ("MMLongBench", "data/golden/mmlongbench-v1.yaml")]
LLM_MODEL = "gemma3:4b"  # local Ollama; cloud-via-ollama preference, worked in Phase 2


def leg(cat: str) -> str:
    return "visual" if cat in HYBRID else "text"


def summarize(name: str, rows: list[tuple[str, str, str]]) -> str:
    # rows: (true_category, regex_leg, llm_leg|"-")
    incorp = [r for r in rows if r[0] != "out_of_corpus"]
    need_visual = [r for r in incorp if r[0] in HYBRID]
    n_in, n_nv = len(incorp), len(need_visual)
    out = [f"\n## {name}  (in-corpus: {n_in}, of which need-visual: {n_nv})"]
    if n_nv == 0:
        out.append("  No figure/table/multi_hop queries — routing can't help here.")
        return "\n".join(out)
    for label, idx in (("regex classify_query", 1), (f"LLM {LLM_MODEL}", 2)):
        misrouted = [r for r in need_visual if r[idx] == "text"]
        if any(r[idx] == "-" for r in need_visual):
            out.append(f"  {label:24s}: (not run)")
            continue
        miss = len(misrouted)
        out.append(
            f"  {label:24s}: {miss}/{n_nv} need-visual queries sent to TEXT "
            f"= {miss / n_nv:.0%} misrouted  -> up to {miss / n_in:.0%} of "
            f"in-corpus accuracy left on the table"
        )
    return "\n".join(out)


async def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # --- regex + oracle: instant, keyless, decisive on its own ---
    per_golden: dict[str, list[tuple[str, str, str]]] = {}
    head = ["# Routing headroom probe", ""]
    for name, path in GOLDENS:
        gs = load_golden_set(Path(path))
        rows = [(q.category, leg(classify_query(q.text)), "-") for q in gs.queries]
        per_golden[name] = rows
        cats: dict[str, int] = {}
        for q in gs.queries:
            cats[q.category] = cats.get(q.category, 0) + 1
        head.append(f"{name}: {dict(sorted(cats.items()))}")
    report = "\n".join(head)
    for name, _ in GOLDENS:
        report += "\n" + summarize(name, per_golden[name])
    report += "\n\n(LLM pass running next; regex result above is already decisive.)\n"
    print(report, flush=True)
    open("data/eval/runs/routing-probe.md", "w", encoding="utf-8").write(report)

    # --- optional LLM classifier pass: cheap (short prompts, no ColQwen2) ---
    try:
        from src.llm.ollama_chat import OllamaChatClient
        from src.prompts.loader import load_prompt_by_name
        from src.rag.retrievers.classifier_llm import LLMQueryClassifier

        clf = LLMQueryClassifier(
            llm=OllamaChatClient(base_url="http://localhost:11434"),
            model=LLM_MODEL,
            prompt=load_prompt_by_name("classify_query"),
        )
        for name, path in GOLDENS:
            gs = load_golden_set(Path(path))
            new: list[tuple[str, str, str]] = []
            for (cat, rgx, _), q in zip(per_golden[name], gs.queries, strict=True):
                try:
                    llm_cat = await clf.classify(q.text)
                    new.append((cat, rgx, leg(llm_cat)))
                except Exception as e:  # noqa: BLE001 - keep going, mark unknown
                    print(f"  classify fail ({type(e).__name__}) on {q.query_id}", flush=True)
                    new.append((cat, rgx, "text"))  # safe-default leg
            per_golden[name] = new
            print(f"LLM pass done: {name}", flush=True)

        final = "# Routing headroom probe (regex + LLM vs oracle)\n" + "\n".join(head[2:])
        for name, _ in GOLDENS:
            final += "\n" + summarize(name, per_golden[name])
        print("\n===FINAL===\n" + final, flush=True)
        open("data/eval/runs/routing-probe.md", "w", encoding="utf-8").write(final)
    except Exception as e:  # noqa: BLE001 - regex verdict already emitted + saved
        print(f"\nLLM pass skipped ({type(e).__name__}: {e}); regex verdict stands.")


if __name__ == "__main__":
    asyncio.run(main())
