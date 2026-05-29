"""Fair-correctness judge over the post-retrieval failures (Bet 3: question the premise).

The official MMLongBench three-stage protocol scores with a strict extractor +
rule-based match. The swarm found a large chunk of "failures" are the model
being right but mis-scored (verbose wrapping, near-synonym, format). And the
public leaderboard (whole-document context, GPT-4o extractor) puts SOTA at ~0.62
while a leaderboard-class model (qwen3-vl-235b) scored only ~0.46 on this repo's
RAG+strict-subset setup. So: how much of the gap is the SCORER, not the model?

This asks a strong free judge (default openai/gpt-oss-120b:free) a fair question
for each post-retrieval failure (gold page WAS fed, official score 0): "ignoring
formatting/phrasing/extra words, is the model's answer factually correct given
the gold?" The count of YES is the harness's false-negative rate on the
gold-present failures — the size of the measurement artifact.

NOT authoring ground truth: the judge compares the model's free text to the
HUMAN gold answer; it only relaxes the strict string match, it does not invent
the gold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scripts.experiments._openrouter_client import build_openrouter_client
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import Message

_PROMPT = (
    "You are grading a question-answering system fairly. You are given a QUESTION, "
    "the human GOLD answer, and the SYSTEM answer.\n"
    "Decide: ignoring formatting, phrasing, extra words, units, and surrounding "
    "prose, is the SYSTEM answer factually correct and equivalent to the GOLD "
    "answer?\n"
    "- Treat a correct value/entity wrapped in a sentence as correct.\n"
    "- Treat a near-synonym or paraphrase that means the same thing as correct.\n"
    "- A genuinely different value/entity, a refusal, or a missing answer is NOT "
    "correct.\n"
    "Reply with exactly one word: YES or NO."
)


async def _judge_one(
    client: OpenRouterClient, model: str, q: str, gold: str, ans: str, *, max_attempts: int = 4
) -> str | None:
    user = f"QUESTION: {q}\nGOLD: {gold}\nSYSTEM: {ans}\n\nIs the SYSTEM answer factually correct? YES or NO."
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.chat(
                [Message(role="system", content=_PROMPT), Message(role="user", content=user)],
                model=model,
                temperature=0.0,
                max_tokens=8,
            )
            txt = (resp.text or "").strip().upper()
            if "YES" in txt:
                return "YES"
            if "NO" in txt:
                return "NO"
            return None
        except Exception:
            if attempt == max_attempts:
                return None
            await asyncio.sleep(min(2.0 * attempt, 12.0))
    return None


async def run(args: argparse.Namespace) -> int:
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    client = build_openrouter_client(timeout=args.timeout)
    cache: dict[str, str] = {}
    if args.cache and args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    yes = no = unparsed = 0
    by_cat: dict[str, list[int]] = {}
    for f in failures:
        qid = f["qid"]
        verdict: str | None
        if qid in cache:
            verdict = cache[qid]
        else:
            verdict = await _judge_one(
                client, args.model, f["query"], str(f["gold"]), str(f["model_answer"])
            )
            if verdict:
                cache[qid] = verdict
                if args.cache:
                    args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        cat = f.get("category", "")
        bucket = by_cat.setdefault(cat, [0, 0])
        if verdict == "YES":
            yes += 1
            bucket[0] += 1
        elif verdict == "NO":
            no += 1
        else:
            unparsed += 1
        bucket[1] += 1
        if args.verbose:
            print(
                f"  {qid:<34} gold={str(f['gold'])[:22]!r:<24} -> {verdict}  ({str(f['model_answer'])[:40]!r})",
                flush=True,
            )

    n = yes + no
    print(
        f"\nFair-judge ({args.model}) over {len(failures)} post-retrieval failures (official score 0):"
    )
    print(f"  judged={n}  unparsed={unparsed}")
    print(
        f"  actually CORRECT (harness false-negative): {yes}/{n}  ({yes / n:.0%})"
        if n
        else "  none judged"
    )
    print("  by category (correct / total):")
    for c in sorted(by_cat):
        ok, tot = by_cat[c]
        print(f"    {c:<10} {ok}/{tot}")
    if args.answerable_total and args.already_correct is not None:
        fair = (args.already_correct + yes) / args.answerable_total
        strict = args.already_correct / args.answerable_total
        print(
            f"\n  implied answerable accuracy: strict={strict:.3f} -> fair={fair:.3f} (+{fair - strict:.3f})"
        )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--failures",
        type=Path,
        default=Path("docs/research/2026-05-29-agenda/postret_failures.json"),
    )
    ap.add_argument("--model", default="openai/gpt-oss-120b:free")
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/fair_judge_cache.json"))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--answerable-total", type=int, default=None, help="for the implied-accuracy line"
    )
    ap.add_argument(
        "--already-correct", type=int, default=None, help="answerable queries already scored >0"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
