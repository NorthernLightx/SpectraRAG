"""Experiment: vision generator vs text generator on figure/table queries.

Asks: does sending page images to a VLM (Qwen3-VL-32b-instruct) actually beat
a text-only LLM (gpt-4o-mini) on golden v3's figure / table queries — or does
the page TEXT alone (caption + nearby paragraphs that PyMuPDF extracts)
already give the text generator everything it needs?

Method:
- Filter golden v3 to category in {figure, table} with relevant_pages set
  (or derivable from chunk-id format `paper::pN::cM`).
- For each query, both generators get the SAME text context — full text of
  the relevant pages via PyMuPDF. Vision additionally receives the rendered
  page PNG (data/pages/<paper>/<paper>_pN.png) as a content block. So vision
  has STRICTLY MORE information; if it doesn't outperform, the image isn't
  adding value for that query.
- Same answer prompt (src/prompts/library/answer.yaml) for both.
- LLMJudge scores faithfulness + answer_relevance for each answer with
  gpt-4o-mini as the judge. Run JSON written to data/eval/runs/.

Cost: ~$0.05 across the in-corpus figure/table queries (text-side calls
~$0.001 each, vision-side ~$0.0003 with Qwen3-VL-32b's compact image
tokenisation, judges ~$0.001 each).

Caveat — the experiment compares GENERATION quality given fixed context.
It does NOT test retrieval (golden truth fills that role). Conclusions
about "vision wins" only apply to the generation step.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Loads .env for RAG_OPENROUTER_API_KEY
import src  # noqa: F401
from src.eval.golden_set import load_golden_set
from src.eval.judges import JudgeOutput, LLMJudge
from src.ingestion.pdf import extract_pages
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import Message
from src.prompts.loader import load_prompt_by_name
from src.types import RetrievalResult

# cp1252-safe stdout (matches configure_logging) so model output with π/≥/etc.
# doesn't crash mid-print on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_URL = "https://openrouter.ai/api/v1/chat/completions"
_PAGE_RE = re.compile(r"::p(\d+)")


def _pages_for_query(query_relevant_pages: list[int], chunk_ids: list[str]) -> list[int]:
    """Use relevant_pages if set; else derive from chunk-id format `paper::pN::cM`."""
    if query_relevant_pages:
        return sorted(set(query_relevant_pages))
    pages: set[int] = set()
    for cid in chunk_ids:
        m = _PAGE_RE.search(cid)
        if m:
            pages.add(int(m.group(1)))
    return sorted(pages)


def _encode_image(path: Path) -> str:
    return f"data:image/png;base64,{base64.standard_b64encode(path.read_bytes()).decode()}"


@dataclass
class GenOutcome:
    text: str
    tokens_in: int
    tokens_out: int
    elapsed_ms: int
    error: str | None = None


async def _text_generate(
    client: OpenRouterClient,
    *,
    model: str,
    system: str | None,
    user: str,
) -> GenOutcome:
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=user))
    t0 = time.monotonic()
    try:
        resp = await client.chat(messages=messages, model=model, temperature=0.2, max_tokens=400)
    except Exception as exc:
        return GenOutcome(
            text="",
            tokens_in=0,
            tokens_out=0,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )
    return GenOutcome(
        text=resp.text,
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


async def _vision_generate(
    *,
    api_key: str,
    model: str,
    system: str | None,
    user: str,
    image_paths: list[Path],
) -> GenOutcome:
    """OpenAI-compat content blocks for vision. Bypasses OpenRouterClient
    (which only takes string-content messages today)."""
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 400}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as exc:
        return GenOutcome(
            text="",
            tokens_in=0,
            tokens_out=0,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except Exception as exc:
        return GenOutcome(
            text="",
            tokens_in=0,
            tokens_out=0,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )
    usage = data.get("usage", {}) or {}
    return GenOutcome(
        text=data["choices"][0]["message"]["content"],
        tokens_in=int(usage.get("prompt_tokens", 0)),
        tokens_out=int(usage.get("completion_tokens", 0)),
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


@dataclass
class PerQuery:
    query_id: str
    paper_id: str
    category: str
    query: str
    pages: list[int]
    text: GenOutcome
    vision: GenOutcome
    text_faith: float | None = None
    text_rel: float | None = None
    vision_faith: float | None = None
    vision_rel: float | None = None
    text_faith_rationale: str = ""
    vision_faith_rationale: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "paper_id": self.paper_id,
            "category": self.category,
            "query": self.query,
            "pages": self.pages,
            "text": {
                "answer": self.text.text,
                "tokens_in": self.text.tokens_in,
                "tokens_out": self.text.tokens_out,
                "elapsed_ms": self.text.elapsed_ms,
                "error": self.text.error,
            },
            "vision": {
                "answer": self.vision.text,
                "tokens_in": self.vision.tokens_in,
                "tokens_out": self.vision.tokens_out,
                "elapsed_ms": self.vision.elapsed_ms,
                "error": self.vision.error,
            },
            "judges": {
                "text": {
                    "faithfulness": self.text_faith,
                    "answer_relevance": self.text_rel,
                    "faithfulness_rationale": self.text_faith_rationale,
                },
                "vision": {
                    "faithfulness": self.vision_faith,
                    "answer_relevance": self.vision_rel,
                    "faithfulness_rationale": self.vision_faith_rationale,
                },
            },
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v3.yaml"))
    parser.add_argument("--text-model", default="openai/gpt-4o-mini")
    parser.add_argument("--vision-model", default="qwen/qwen3-vl-32b-instruct")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap number of queries (0 = all eligible)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/runs/exp_vision_vs_text.json")
    )
    args = parser.parse_args()

    api_key = os.environ.get("RAG_OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("RAG_OPENROUTER_API_KEY not set; check .env")

    answer_prompt = load_prompt_by_name("answer")
    faith_prompt = load_prompt_by_name("judge_faithfulness")
    rel_prompt = load_prompt_by_name("judge_answer_relevance")
    ctx_prec_prompt = load_prompt_by_name("judge_context_precision")  # required by LLMJudge ctor

    text_client = OpenRouterClient(api_key=api_key)
    judge_client = OpenRouterClient(api_key=api_key)
    judge = LLMJudge(
        llm=judge_client,
        model=args.judge_model,
        faithfulness_prompt=faith_prompt,
        answer_relevance_prompt=rel_prompt,
        context_precision_prompt=ctx_prec_prompt,
    )

    golden = load_golden_set(args.golden)
    eligible = []
    for q in golden.queries:
        if q.category not in {"figure", "table"}:
            continue
        pages = _pages_for_query(q.relevant_pages, q.relevant_chunk_ids)
        if not pages:
            continue
        # Need both PDF and at least one rendered page image
        pdf = Path(f"data/papers/{q.paper_id}.pdf")
        if not pdf.exists():
            continue
        renders = [Path(f"data/pages/{q.paper_id}/{q.paper_id}_p{p}.png") for p in pages]
        renders = [p for p in renders if p.exists()]
        if not renders:
            continue
        eligible.append((q, pages, pdf, renders))

    if args.limit and args.limit > 0:
        eligible = eligible[: args.limit]

    print(f"Running on {len(eligible)} eligible figure/table queries")
    print(f"Text:   {args.text_model}")
    print(f"Vision: {args.vision_model}")
    print(f"Judge:  {args.judge_model}")
    print()

    results: list[PerQuery] = []
    for q, pages, pdf, renders in eligible:
        # Get text of relevant pages — use PyMuPDF directly
        all_pages = extract_pages(q.paper_id, pdf)
        page_text = "\n\n".join(
            f"[{q.paper_id}::p{p.page_number}] {p.text}"
            for p in all_pages
            if p.page_number in pages
        )
        if not page_text.strip():
            print(f"[skip] {q.query_id}: no text on relevant pages")
            continue

        system, user = answer_prompt.render(query=q.text, context=page_text)

        # Generate both ways
        print(f"=== {q.query_id} (p{pages}, {q.category}) — {q.text[:60]}...")
        text_out = await _text_generate(
            text_client, model=args.text_model, system=system, user=user
        )
        vis_out = await _vision_generate(
            api_key=api_key,
            model=args.vision_model,
            system=system,
            user=user,
            image_paths=renders,
        )

        # Judge — use a single fake RetrievalResult holding the page text so
        # LLMJudge.faithfulness sees the full context.
        retrieved_for_judge = [
            RetrievalResult(
                chunk_id=f"{q.paper_id}::p{p}::page",
                paper_id=q.paper_id,
                score=1.0,
                text=next(pp.text for pp in all_pages if pp.page_number == p),
                page_numbers=[p],
                source="pipeline",
            )
            for p in pages
            if any(pp.page_number == p for pp in all_pages)
        ]

        text_faith: JudgeOutput | None = None
        vis_faith: JudgeOutput | None = None
        text_rel: JudgeOutput | None = None
        vis_rel: JudgeOutput | None = None

        if not text_out.error:
            text_faith = await judge.faithfulness(
                query=q.text, answer=text_out.text, retrieved=retrieved_for_judge
            )
            text_rel = await judge.answer_relevance(query=q.text, answer=text_out.text)
        if not vis_out.error:
            vis_faith = await judge.faithfulness(
                query=q.text, answer=vis_out.text, retrieved=retrieved_for_judge
            )
            vis_rel = await judge.answer_relevance(query=q.text, answer=vis_out.text)

        per_q = PerQuery(
            query_id=q.query_id,
            paper_id=q.paper_id,
            category=q.category,
            query=q.text,
            pages=pages,
            text=text_out,
            vision=vis_out,
            text_faith=text_faith.score if text_faith else None,
            text_rel=text_rel.score if text_rel else None,
            vision_faith=vis_faith.score if vis_faith else None,
            vision_rel=vis_rel.score if vis_rel else None,
            text_faith_rationale=text_faith.rationale if text_faith else "",
            vision_faith_rationale=vis_faith.rationale if vis_faith else "",
        )
        results.append(per_q)

        # Live progress line
        print(
            f"  text:   faith={per_q.text_faith}, rel={per_q.text_rel}, "
            f"lat={text_out.elapsed_ms}ms, in/out={text_out.tokens_in}/{text_out.tokens_out}"
            + (f"  ERROR: {text_out.error}" if text_out.error else "")
        )
        print(
            f"  vision: faith={per_q.vision_faith}, rel={per_q.vision_rel}, "
            f"lat={vis_out.elapsed_ms}ms, in/out={vis_out.tokens_in}/{vis_out.tokens_out}"
            + (f"  ERROR: {vis_out.error}" if vis_out.error else "")
        )
        print()

    # Aggregate
    if results:

        def avg(vals: list[float | None]) -> float | None:
            xs = [v for v in vals if v is not None]
            return sum(xs) / len(xs) if xs else None

        tf = avg([r.text_faith for r in results])
        tr = avg([r.text_rel for r in results])
        vf = avg([r.vision_faith for r in results])
        vr = avg([r.vision_rel for r in results])
        tlat = sum(r.text.elapsed_ms for r in results) / len(results)
        vlat = sum(r.vision.elapsed_ms for r in results) / len(results)
        tin = sum(r.text.tokens_in for r in results)
        tout = sum(r.text.tokens_out for r in results)
        vin = sum(r.vision.tokens_in for r in results)
        vout = sum(r.vision.tokens_out for r in results)

        print("=" * 60)
        print(f"Aggregate over {len(results)} queries")
        print("-" * 60)
        print("            faithfulness  answer_rel.   p_lat(ms)  tokens(in/out)")
        print(f"text    {args.text_model:>32s}  ", end="")
        print(f"  {tf or 0:.3f}        {tr or 0:.3f}        {tlat:.0f}      {tin}/{tout}")
        print(f"vision  {args.vision_model:>32s}  ", end="")
        print(f"  {vf or 0:.3f}        {vr or 0:.3f}        {vlat:.0f}      {vin}/{vout}")
        print()

        # Per-query winner column
        print("Per-query (Δ = vision - text on faithfulness):")
        for r in results:
            tf_v = r.text_faith if r.text_faith is not None else 0.0
            vf_v = r.vision_faith if r.vision_faith is not None else 0.0
            delta = vf_v - tf_v
            sign = "+" if delta > 0 else ("=" if delta == 0 else "")
            print(f"  {r.query_id:42s}  text={tf_v:.2f}  vision={vf_v:.2f}  Δ={sign}{delta:+.2f}")

    # Save JSON for the record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
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
                "per_query": [r.to_json() for r in results],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
