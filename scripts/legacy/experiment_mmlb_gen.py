"""Multi-modal vs text-only GENERATION on MMLongBench-Doc.

Question: when both generators get identical context (the chunks from gold-evidence
pages + the rendered page image as a content block for the multi-modal one), does
vision actually beat text on a corpus where figures encode info as pixels (charts,
screenshots, image-only diagrams)?

The earlier golden v3 experiment (commit 373cccc) showed null because PyMuPDF
text-layer extraction was already sufficient on modern arXiv preprints. MMLongBench
is the test on a corpus where text extraction is genuinely inadequate — Pew
Research charts, screenshots, brochures.

Anti-overfitting design:
- All in-corpus queries (factual + figure + table + multi_hop), not just figure.
  If vision wins uniformly across categories that's suspicious; if it wins on
  figure/table/multi_hop and ties on factual, that's the right signal.
- Identical context for both generators — gold-evidence page text + image.
- Two scoring channels: LLM judge (faithfulness, answer_relevance) AND
  programmatic gold-answer match (case-insensitive substring). The programmatic
  channel defuses judge bias toward answer style.
- All metrics reported, no cherry-picking.

Cost: ~$0.50 across 109 in-corpus queries x (1 gen text + 1 gen vision +
2 judges) at ~$0.001/call avg. Wall clock ~30-40 min.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

import src  # noqa: F401  -- loads .env
from src.eval.golden_set import load_golden_set
from src.eval.judges import LLMJudge
from src.ingestion.pdf import extract_pages
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import Message
from src.prompts.loader import load_prompt_by_name
from src.types import RetrievalResult

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_URL = "https://openrouter.ai/api/v1/chat/completions"


def _encode_image(path: Path) -> str:
    return f"data:image/png;base64,{base64.standard_b64encode(path.read_bytes()).decode()}"


def _gold_answer_in(answer: str, gold: str) -> bool:
    """Case-insensitive substring match of gold in answer. Bypasses judge bias."""
    if not gold or not answer:
        return False
    return gold.strip().lower() in answer.lower()


@dataclass
class Outcome:
    text: str
    tokens_in: int
    tokens_out: int
    elapsed_ms: int
    error: str | None = None


async def _text_gen(client: OpenRouterClient, model: str, system: str | None, user: str) -> Outcome:
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=user))
    t0 = time.monotonic()
    try:
        resp = await client.chat(messages=messages, model=model, temperature=0.2, max_tokens=400)
    except Exception as exc:
        return Outcome(
            text="",
            tokens_in=0,
            tokens_out=0,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )
    return Outcome(
        text=resp.text,
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def _vision_gen(
    api_key: str, model: str, system: str | None, user: str, images: list[Path]
) -> Outcome:
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for p in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 400}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as exc:
        return Outcome(
            text="",
            tokens_in=0,
            tokens_out=0,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except Exception as exc:
        return Outcome(
            text="",
            tokens_in=0,
            tokens_out=0,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )
    usage = data.get("usage", {}) or {}
    return Outcome(
        text=data["choices"][0]["message"]["content"],
        tokens_in=int(usage.get("prompt_tokens", 0)),
        tokens_out=int(usage.get("completion_tokens", 0)),
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    p.add_argument("--pdfs", type=Path, default=Path("data/mmlongbench/documents"))
    p.add_argument("--pages", type=Path, default=Path("data/pages"))
    p.add_argument("--text-model", default="openai/gpt-4o-mini")
    p.add_argument("--vision-model", default="qwen/qwen3-vl-32b-instruct")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--limit", type=int, default=0, help="Cap query count (0 = all eligible).")
    p.add_argument(
        "--query-ids",
        type=str,
        default="",
        help=(
            "Comma-separated query_ids to filter to (e.g. for a focused smoke test "
            "before running the full eval). Empty = no filter."
        ),
    )
    p.add_argument(
        "--exclude-ooc",
        action="store_true",
        default=True,
        help="Skip out_of_corpus queries (default: True; gen comparison needs answerable Qs).",
    )
    p.add_argument("--output", type=Path, default=Path("data/eval/runs/exp_mmlb_gen.json"))
    args = p.parse_args()

    api_key = os.environ.get("RAG_OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("RAG_OPENROUTER_API_KEY not set")

    answer_prompt = load_prompt_by_name("answer")
    faith_prompt = load_prompt_by_name("judge_faithfulness")
    rel_prompt = load_prompt_by_name("judge_answer_relevance")
    cprec_prompt = load_prompt_by_name("judge_context_precision")  # required by ctor

    text_client = OpenRouterClient(api_key=api_key)
    judge = LLMJudge(
        llm=OpenRouterClient(api_key=api_key),
        model=args.judge_model,
        faithfulness_prompt=faith_prompt,
        answer_relevance_prompt=rel_prompt,
        context_precision_prompt=cprec_prompt,
    )

    golden = load_golden_set(args.golden)
    qid_filter = {x.strip() for x in args.query_ids.split(",") if x.strip()}
    eligible = []
    for q in golden.queries:
        if qid_filter and q.query_id not in qid_filter:
            continue
        if args.exclude_ooc and q.category == "out_of_corpus":
            continue
        if not q.relevant_pages:
            continue
        pdf = args.pdfs / f"{q.paper_id}.pdf"
        if not pdf.exists():
            continue
        page_imgs = [args.pages / q.paper_id / f"{q.paper_id}_p{p}.png" for p in q.relevant_pages]
        page_imgs = [p for p in page_imgs if p.exists()]
        if not page_imgs:
            continue
        eligible.append((q, pdf, page_imgs))

    if args.limit and args.limit > 0:
        eligible = eligible[: args.limit]

    print(f"Running on {len(eligible)} eligible queries")
    print(f"Text:    {args.text_model}")
    print(f"Vision:  {args.vision_model}")
    print(f"Judge:   {args.judge_model}")
    print()

    per_query: list[dict[str, Any]] = []
    cat_n: Counter[str] = Counter()
    cat_text_match: defaultdict[str, list[bool]] = defaultdict(list)
    cat_vision_match: defaultdict[str, list[bool]] = defaultdict(list)
    cat_text_faith: defaultdict[str, list[float]] = defaultdict(list)
    cat_vision_faith: defaultdict[str, list[float]] = defaultdict(list)
    cat_text_rel: defaultdict[str, list[float]] = defaultdict(list)
    cat_vision_rel: defaultdict[str, list[float]] = defaultdict(list)

    for idx, (q, pdf, page_imgs) in enumerate(eligible):
        # Build text context = full PyMuPDF text of the gold pages
        all_pages = extract_pages(q.paper_id, pdf)
        gold_pages = {p.page_number for p in all_pages if p.page_number in q.relevant_pages}
        ctx_text = "\n\n".join(
            f"[{q.paper_id}::p{p.page_number}::page] {p.text}"
            for p in all_pages
            if p.page_number in gold_pages
        )
        if not ctx_text.strip():
            print(f"[skip] {q.query_id}: empty page text")
            continue

        system, user = answer_prompt.render(query=q.text, context=ctx_text)
        gold_answer = q.expected_facts[0] if q.expected_facts else ""

        # Generate both ways
        t_out = await _text_gen(text_client, args.text_model, system, user)
        v_out = await _vision_gen(api_key, args.vision_model, system, user, page_imgs)

        # Programmatic gold-answer match
        text_match = _gold_answer_in(t_out.text, gold_answer) and not t_out.error
        vis_match = _gold_answer_in(v_out.text, gold_answer) and not v_out.error

        # LLM judges
        retrieved_for_judge = [
            RetrievalResult(
                chunk_id=f"{q.paper_id}::p{p.page_number}::page",
                paper_id=q.paper_id,
                score=1.0,
                text=p.text,
                page_numbers=[p.page_number],
                source="pipeline",
            )
            for p in all_pages
            if p.page_number in gold_pages
        ]
        text_faith = await judge.faithfulness(
            query=q.text, answer=t_out.text, retrieved=retrieved_for_judge
        )
        text_rel = await judge.answer_relevance(query=q.text, answer=t_out.text)
        vis_faith = await judge.faithfulness(
            query=q.text, answer=v_out.text, retrieved=retrieved_for_judge
        )
        vis_rel = await judge.answer_relevance(query=q.text, answer=v_out.text)

        cat: str = q.category
        cat_n[cat] += 1
        cat_text_match[cat].append(text_match)
        cat_vision_match[cat].append(vis_match)
        cat_text_faith[cat].append(text_faith.score)
        cat_vision_faith[cat].append(vis_faith.score)
        cat_text_rel[cat].append(text_rel.score)
        cat_vision_rel[cat].append(vis_rel.score)

        print(
            f"  [{idx + 1:3d}/{len(eligible)}] {q.query_id:38s} cat={cat:10s} "
            f"gold={'T' if text_match else '.'}{'V' if vis_match else '.'}  "
            f"faith t={text_faith.score:.2f}/v={vis_faith.score:.2f}  "
            f"rel t={text_rel.score:.2f}/v={vis_rel.score:.2f}"
        )

        per_query.append(
            {
                "query_id": q.query_id,
                "paper_id": q.paper_id,
                "category": cat,
                "query": q.text,
                "gold": gold_answer,
                "pages": q.relevant_pages,
                "text": {
                    "answer": t_out.text,
                    "tokens_in": t_out.tokens_in,
                    "tokens_out": t_out.tokens_out,
                    "elapsed_ms": t_out.elapsed_ms,
                    "error": t_out.error,
                    "gold_match": text_match,
                    "faithfulness": text_faith.score,
                    "answer_relevance": text_rel.score,
                },
                "vision": {
                    "answer": v_out.text,
                    "tokens_in": v_out.tokens_in,
                    "tokens_out": v_out.tokens_out,
                    "elapsed_ms": v_out.elapsed_ms,
                    "error": v_out.error,
                    "gold_match": vis_match,
                    "faithfulness": vis_faith.score,
                    "answer_relevance": vis_rel.score,
                },
            }
        )

    # Aggregate
    def avg(xs: list[float] | list[bool]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(per_query)} queries  ({args.text_model} vs {args.vision_model})")
    print("=" * 78)
    all_t_match = [r["text"]["gold_match"] for r in per_query]
    all_v_match = [r["vision"]["gold_match"] for r in per_query]
    all_t_faith = [r["text"]["faithfulness"] for r in per_query]
    all_v_faith = [r["vision"]["faithfulness"] for r in per_query]
    all_t_rel = [r["text"]["answer_relevance"] for r in per_query]
    all_v_rel = [r["vision"]["answer_relevance"] for r in per_query]
    print(
        f"  gold-answer match    text={avg(all_t_match):.4f}  vision={avg(all_v_match):.4f}  "
        f"Δ={avg(all_v_match) - avg(all_t_match):+.4f}"
    )
    print(
        f"  faithfulness         text={avg(all_t_faith):.4f}  vision={avg(all_v_faith):.4f}  "
        f"Δ={avg(all_v_faith) - avg(all_t_faith):+.4f}"
    )
    print(
        f"  answer_relevance     text={avg(all_t_rel):.4f}  vision={avg(all_v_rel):.4f}  "
        f"Δ={avg(all_v_rel) - avg(all_t_rel):+.4f}"
    )
    print()
    print("PER-CATEGORY")
    print(
        f"  {'category':12s}  n   "
        f"{'gold-match (t/v/Δ)':>27s}  {'faith (t/v/Δ)':>22s}  {'rel (t/v/Δ)':>22s}"
    )
    for cat in sorted(cat_n.keys()):
        n = cat_n[cat]
        tm = avg(cat_text_match[cat])
        vm = avg(cat_vision_match[cat])
        tf = avg(cat_text_faith[cat])
        vf = avg(cat_vision_faith[cat])
        tr = avg(cat_text_rel[cat])
        vr = avg(cat_vision_rel[cat])
        print(
            f"  {cat:12s}  {n:3d}   "
            f"{tm:.2f}/{vm:.2f}/{vm - tm:+.2f}        "
            f"{tf:.2f}/{vf:.2f}/{vf - tf:+.2f}     "
            f"{tr:.2f}/{vr:.2f}/{vr - tr:+.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "text_model": args.text_model,
                    "vision_model": args.vision_model,
                    "judge_model": args.judge_model,
                    "golden": str(args.golden),
                    "answer_prompt_version": answer_prompt.version,
                    "judge_faith_version": faith_prompt.version,
                    "judge_rel_version": rel_prompt.version,
                },
                "per_query": per_query,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
