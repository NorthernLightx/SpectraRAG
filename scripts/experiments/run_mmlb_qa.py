"""Generate an MMLongBench-Doc QA run with a cloud VLM, for score_mmlb_qa.py.

This is the GENERATION half of the official three-stage protocol (see
scripts/experiments/score_mmlb_qa.py for stages 2-3). It is fully decoupled
from retrieval: retrieval is run separately on the GPU box by
scripts/experiments/diagnose_depth50_run.py, which dumps a depth-50 JSON; this
harness CONSUMES that JSON and only does generation + output. No retriever,
embedder, or ColQwen2 is loaded here, so it triggers zero local GPU/VRAM work
(it is safe to run alongside a depth-50 retrieval diagnostic on the same card).

  PIPELINE
  --------
  retrieval JSON (depth-50, off-GPU dump)
      per_query[i] = {query_id, category, text_top50, visual_top50, fused_top50}
      each *_top50 a list of chunk-ids like `paper::pN::cM` or `paper::pN::page`
                                |
                                v  this harness
  for each query: fused_top50 -> top-K UNIQUE (paper, page) pages (rank order)
      -> load data/pages/<paper>/<paper>_p<N>.png for each page
      -> vision prompt (question + page PNGs as Ollama image blocks)
      -> qwen3-vl:235b-cloud via Ollama /api/chat
      -> answer text
                                |
                                v
  run JSON  per_query[i] = {query_id, ..., vision: {answer, ...}}
      consumed by score_mmlb_qa.py --answer-field vision.answer

  WHY OLLAMA REST DIRECTLY (not OllamaChatClient)
  -----------------------------------------------
  src/llm/ollama_chat.py:OllamaChatClient.chat() deliberately drops the
  `images` arg (it never bridged Ollama's per-message base64 image field). This
  harness needs vision, so it POSTs /api/chat itself with the documented Ollama
  contract: each message carries `images: [<base64-png>, ...]` (NOT the
  OpenAI-compat `image_url` content blocks that src/llm/openrouter.py builds —
  that schema is OpenRouter-only). Verified against qwen3-vl:235b-cloud
  2026-05-25: a real page PNG returned the correct gold answer in ~12s.

  qwen3-vl:235b-cloud is a REMOTE cloud model exposed through local Ollama, so
  the 235B weights never touch local VRAM. Routed through Ollama by project
  convention (NOT OpenRouter).

  PAGE MAPPING is identical to the retrieval scorer: the same `::p(\\d+)` regex
  and rank-order dedup as scripts/rescore_mmlb_pages.py (imported, not
  reimplemented, so a page identity here is the same tuple retrieval is scored
  on). A chunk-id with no `::pN::` segment is skipped.

  GOLD lives in data/golden/mmlongbench-v1.yaml — all 149 queries INCLUDING the
  36 unanswerable ones; they are needed for the official F1 (the negative
  class). The MACHINE NEVER AUTHORS GROUND TRUTH: this harness only reads the
  human gold short answer (expected_facts[0]) into the run for convenience and
  generates a model answer. The refusal phrasing in the system prompt lets an
  unanswerable query be answered-as-refusal so the stage-2 extractor maps it to
  "Not answerable".

  RESUMABLE: a cache JSON keyed by model+top_k+query_id stores each answer as it
  lands and is written after every query, so a long 149-query run (cloud
  latency ~10-15s each => ~25-40 min) survives interruption / rate limits. Rerun
  with the same --cache to continue.

Usage (once the real depth-50 JSON lands on the GPU box):

    .venv/Scripts/python.exe -m scripts.experiments.run_mmlb_qa \\
        --retrieval data/eval/runs/depth50-<ts>/depth50.json \\
        --golden data/golden/mmlongbench-v1.yaml \\
        --out data/eval/runs/exp_mmlb_gen_cloud.json \\
        --cache data/eval/runs/mmlb_gen_cloud_cache.json \\
        --top-k 5

then score it:

    .venv/Scripts/python.exe -m scripts.experiments.score_mmlb_qa \\
        --run data/eval/runs/exp_mmlb_gen_cloud.json \\
        --golden data/golden/mmlongbench-v1.yaml \\
        --answer-field vision.answer \\
        --extractor-model gemma3:4b \\
        --cache data/eval/runs/mmlb_qa_extract_cache.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.experiments._openrouter_client import build_openrouter_client

# Reuse the EXACT page-identity logic the retrieval scorer uses, so a (paper,
# page) tuple here is the same one retrieval is graded on. These are private
# helpers, but diagnose_depth50_run.py already imports `rescore` from this same
# module; sharing the page mapping is the point (no second source of truth).
from scripts.rescore_mmlb_pages import Page, _dedup_pages_in_rank
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import Message

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# Pre-rendered page images live at data/pages/<paper_id>/<paper_id>_p<N>.png
# (confirmed on disk 2026-05-25; produced by src.ingestion.visual.render_pages
# in diagnose_depth50_run.py at dpi=150).
_PAGES_ROOT = Path("data/pages")

# Vision generation prompt. Modelled on src/prompts/library/answer.yaml (the
# repo's generator prompt) but adapted for page IMAGES rather than text chunks.
# The refusal phrasing matters: the stage-2 extractor
# (score_mmlb_qa.py:_EXTRACTION_PROMPT) maps "can not be answered from the
# given documents" to the negative class "Not answerable". Saying exactly "Not
# answerable" makes the unanswerable queries answerable-as-refusal, which is
# what the official F1 needs. answer.yaml's "Not stated in the provided
# context." is a different string the extractor would not necessarily collapse
# to the negative class, so we do NOT reuse it verbatim here.
_SYSTEM_PROMPT = (
    "You are a careful research assistant answering a question from the page "
    "images of a document. Use only what is visible in the images.\n"
    "- Read the pages (text, tables, charts, figures) and answer the question "
    "directly and concisely.\n"
    '- If the images do not contain the answer, reply exactly "Not answerable" '
    "and nothing else. Do not guess.\n"
    "- Give the answer itself (a number, a short phrase, or a short list), not "
    "a description of where it appears."
)

_USER_TEMPLATE = "Question: {query}\n\nAnswer using only the {n_pages} page image(s) above."


def _png_path(paper: str, page: int) -> Path:
    return _PAGES_ROOT / paper / f"{paper}_p{page}.png"


def _encode_png(path: Path) -> str:
    """PNG path -> bare base64 (Ollama /api/chat `images` wants raw base64, NOT
    a data: URL — that is the OpenRouter/OpenAI content-block convention)."""
    return base64.standard_b64encode(path.read_bytes()).decode()


def _gold_answer(query: dict[str, Any]) -> str:
    facts = query.get("expected_facts") or [""]
    return str(facts[0])


def select_pages(fused_top50: list[str], top_k: int) -> list[Page]:
    """Top-K unique (paper, page) pages from the fused ranking, in rank order.

    Dedup is rank-order first-appearance (same as the retrieval scorer), then
    truncated to K. Chunk-ids without a `::pN::` segment are dropped by
    _dedup_pages_in_rank.
    """
    return _dedup_pages_in_rank(fused_top50)[:top_k]


async def _chat_vision(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    system: str,
    user: str,
    images_b64: list[str],
    *,
    temperature: float,
    max_tokens: int,
    num_gpu: int | None = None,
    num_ctx: int | None = None,
    max_attempts: int = 5,
) -> tuple[str, int, int]:
    """One vision /api/chat call. Returns (answer_text, tokens_in, tokens_out).

    The page images ride on the user message's `images` field (Ollama's
    contract). Bounded backoff on transport errors and HTTP 5xx — the cloud
    backend can 429/5xx under load over a 149-query run; we retry rather than
    abort the whole run on one transient failure. A persistent failure raises
    so the caller leaves the query UNcached for a later resume.

    `num_gpu` / `num_ctx` are optional Ollama runtime options. They are only
    relevant for a LOCAL model (a cloud model ignores them): num_gpu forces a
    layer count onto the GPU (a high value like 99 = "all layers", overriding
    Ollama's conservative auto-offload that drops a vision model to CPU when it
    can't prove the page-image activations fit); num_ctx caps the context
    window (a smaller window shrinks the KV cache). Both are omitted from the
    payload when None, so the default behavior is unchanged. This is the same
    omit-when-None convention as src/llm/ollama_chat.py.
    """
    options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user, "images": images_b64},
        ],
        "stream": False,
        "options": options,
    }
    url = f"{base_url.rstrip('/')}/api/chat"
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message") or {}
            return (
                message.get("content", "") or "",
                int(data.get("prompt_eval_count", 0) or 0),
                int(data.get("eval_count", 0) or 0),
            )
        except httpx.HTTPStatusError as exc:
            # 4xx other than 429 (e.g. bad request) is not transient — fail fast.
            if exc.response.status_code not in (429,) and exc.response.status_code < 500:
                raise
            last_exc = exc
        except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
        if attempt < max_attempts:
            await asyncio.sleep(min(2.0 * attempt, 15.0))
    assert last_exc is not None  # loop ran >=1 time and only reaches here on failure
    raise last_exc


async def _chat_vision_openrouter(
    client: OpenRouterClient,
    model: str,
    system: str,
    user: str,
    image_paths: list[Path],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    """OpenRouter equivalent of _chat_vision. Same return contract
    (answer_text, tokens_in, tokens_out) so the call site is provider-uniform.

    OpenRouterClient.chat() attaches `images` to the LAST user message as
    OpenAI-compat `image_url` content blocks (data:image/png;base64,...) and
    encodes the PNG paths itself — the opposite convention to Ollama's raw
    base64 `images` field, which is why generation can't share one helper.
    Retry/backoff for transport errors + HTTP 429 lives inside the client
    (tenacity, 6 attempts to 60s); a persistent failure raises here and the
    caller leaves the query uncached for a resume.
    """
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]
    resp = await client.chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        images=image_paths,
    )
    return resp.text, resp.tokens_in, resp.tokens_out


async def run(args: argparse.Namespace) -> int:
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold_by_qid: dict[str, dict[str, Any]] = {q["query_id"]: q for q in golden["queries"]}

    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    per_query_in = retrieval["per_query"] if isinstance(retrieval, dict) else retrieval

    cache: dict[str, dict[str, Any]] = {}
    if args.cache and args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    out_records: list[dict[str, Any]] = []
    skipped_no_gold = 0
    missing_png: set[str] = set()
    n = len(per_query_in)

    # OpenRouter generation reuses the project client (it builds the image_url
    # content blocks and owns retry/backoff); Ollama generation POSTs /api/chat
    # raw because OllamaChatClient deliberately drops images. Built once.
    or_client = (
        build_openrouter_client(timeout=args.timeout) if args.provider == "openrouter" else None
    )

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for i, rec in enumerate(per_query_in):
            qid = rec.get("query_id")
            gold = gold_by_qid.get(qid)
            if gold is None:
                skipped_no_gold += 1
                continue

            question = gold.get("text") or rec.get("query") or ""
            fused = rec.get("fused_top50") or []
            pages = select_pages(fused, args.top_k)

            # Resolve page PNGs; skip any page whose image is missing on disk
            # (record it so the operator sees the gap rather than silently
            # under-feeding the model).
            page_paths: list[tuple[Page, Path]] = []
            for page in pages:
                p = _png_path(page[0], page[1])
                if p.exists():
                    page_paths.append((page, p))
                else:
                    missing_png.add(str(p))

            cache_key = f"{args.model}::k{args.top_k}::{qid}"
            cached = cache.get(cache_key)
            if cached is not None:
                vision = cached
            else:
                if not page_paths:
                    # No usable page image -> the model gets no evidence. Record
                    # an explicit error and an empty answer (scores as a wrong /
                    # empty prediction, never the refusal negative class).
                    vision = {
                        "answer": "",
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "elapsed_ms": 0,
                        "error": "no_page_images",
                    }
                else:
                    user = _USER_TEMPLATE.format(query=question, n_pages=len(page_paths))
                    t0 = time.time()
                    try:
                        if or_client is not None:
                            answer, tin, tout = await _chat_vision_openrouter(
                                or_client,
                                args.model,
                                _SYSTEM_PROMPT,
                                user,
                                [p for _, p in page_paths],
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                            )
                        else:
                            images_b64 = [_encode_png(p) for _, p in page_paths]
                            answer, tin, tout = await _chat_vision(
                                client,
                                args.ollama_url,
                                args.model,
                                _SYSTEM_PROMPT,
                                user,
                                images_b64,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                num_gpu=args.num_gpu,
                                num_ctx=args.num_ctx,
                            )
                    except Exception as exc:
                        # Persistent failure after retries: do NOT cache, so a
                        # rerun with the same --cache retries this query.
                        print(
                            f"  [{i + 1}/{n}] {qid}  GEN FAILED ({type(exc).__name__}: {exc}) "
                            "-- left uncached for resume",
                            flush=True,
                        )
                        out_records.append(
                            {
                                "query_id": qid,
                                "paper_id": gold.get("paper_id"),
                                "category": gold.get("category"),
                                "query": question,
                                "gold": _gold_answer(gold),
                                "pages": [pg[1] for pg in pages],
                                "papers": [pg[0] for pg in pages],
                                "vision": {
                                    "answer": "",
                                    "tokens_in": 0,
                                    "tokens_out": 0,
                                    "elapsed_ms": int((time.time() - t0) * 1000),
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            }
                        )
                        continue
                    vision = {
                        "answer": answer,
                        "tokens_in": tin,
                        "tokens_out": tout,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "error": None,
                    }
                cache[cache_key] = vision
                if args.cache:
                    args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")

            out_records.append(
                {
                    "query_id": qid,
                    "paper_id": gold.get("paper_id"),
                    "category": gold.get("category"),
                    "query": question,
                    "gold": _gold_answer(gold),
                    # `pages` mirrors exp_mmlb_gen_full.json (page numbers only);
                    # `papers` is kept alongside for audit since retrieval is not
                    # paper-scoped and a fused page can come from any paper.
                    "pages": [pg[1] for pg in pages],
                    "papers": [pg[0] for pg in pages],
                    "vision": vision,
                }
            )
            if args.verbose:
                ans_preview = (vision["answer"] or "").replace("\n", " ")[:80]
                print(
                    f"  [{i + 1}/{n}] {qid}  pages={[pg[1] for pg in pages]}  "
                    f"gold={_gold_answer(gold)!r}  ans={ans_preview!r}",
                    flush=True,
                )

    out = {
        "config": {
            "provider": args.provider,
            "vision_model": args.model,
            "ollama_url": args.ollama_url,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_gpu": args.num_gpu,
            "num_ctx": args.num_ctx,
            "retrieval": str(args.retrieval),
            "golden": str(args.golden),
            "answer_prompt": "run_mmlb_qa._SYSTEM_PROMPT",
        },
        "per_query": out_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    answered = sum(1 for r in out_records if (r["vision"]["answer"] or "").strip())
    print(f"\nWrote {args.out}")
    print(
        f"  generated={len(out_records)}  with-answer={answered}  skipped(no gold)={skipped_no_gold}"
    )
    if missing_png:
        print(f"  WARNING: {len(missing_png)} page PNG(s) referenced but missing on disk, e.g.:")
        for missing in sorted(missing_png)[:5]:
            print(f"    {missing}")
    print(
        "\nScore with:\n"
        f"  .venv/Scripts/python.exe -m scripts.experiments.score_mmlb_qa "
        f"--run {args.out} --golden {args.golden} --answer-field vision.answer"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        required=True,
        help="depth-50 retrieval JSON (per_query[i].fused_top50), from diagnose_depth50_run.py",
    )
    parser.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="run JSON to write")
    parser.add_argument(
        "--provider",
        choices=("ollama", "openrouter"),
        default="ollama",
        help="generation backend. 'ollama' POSTs /api/chat raw with base64 images "
        "(default, project convention). 'openrouter' uses OpenRouterClient with "
        "OpenAI-compat image_url blocks (needed when Ollama's daily cloud quota is "
        "exhausted); set --model to an OpenRouter vision id and the key via "
        "RAG_OPENROUTER_API_KEY / .env.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:235b-cloud",
        help="vision model id. For --provider ollama: an Ollama model "
        "(default qwen3-vl:235b-cloud, a cloud model => zero local VRAM). For "
        "--provider openrouter: an OpenRouter vision id, e.g. "
        "google/gemma-4-31b-it:free.",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--top-k", type=int, default=5, help="number of UNIQUE pages from fused_top50 to feed"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="Ollama options.num_gpu: number of model layers to force onto the GPU "
        "(e.g. 99 = all layers). Omitted from the payload when unset. Use to stop a "
        "LOCAL vision model (qwen2.5vl:7b) from auto-offloading to CPU under image load; "
        "a cloud model ignores it.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Ollama options.num_ctx: context-window size (tokens). Omitted from the "
        "payload when unset. A smaller window shrinks the KV cache to claw back VRAM.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-request HTTP timeout (s); cloud vision over many pages can be slow",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="JSON cache of answers (resumable; keyed by model+top_k+query_id)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print each (pages, gold, answer) as it generates"
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
