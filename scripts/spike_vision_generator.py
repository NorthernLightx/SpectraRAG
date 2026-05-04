"""Spike: can a vision-capable OpenRouter model answer figure-grounded queries
when given the page image, where our text-only generator currently fails?

Bypasses OpenRouterClient on purpose — that client's Message.content is a
string, but vision in the OpenAI-compat schema needs a list of content blocks
({"type": "text"}, {"type": "image_url"}). If this spike works we promote the
content-block path into OpenRouterClient + Generator; if not we kill the idea.

Test fixture: three Phase 3.1 figure-grounded queries on paper 2604.28182v1
(q37/q38/q39 from data/golden/v3.yaml) — exactly the kind of question where
text-only retrieval scored weakly per ADR 0007 §"Per-subset" and where the
visual leg is supposed to add value.

Usage:
  $env:RAG_OPENROUTER_API_KEY = "sk-or-v1-..."
  .venv\\Scripts\\python.exe -m scripts.spike_vision_generator

Cost: ~$0.005 across all 12 calls (3 queries x 4 models). Verify on
https://openrouter.ai/activity afterwards.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Loads .env for RAG_OPENROUTER_API_KEY
import src  # noqa: F401

# Match configure_logging's Windows cp1252 fix so model responses containing
# Unicode (π, ≥, etc.) don't crash the printer mid-stream.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_URL = "https://openrouter.ai/api/v1/chat/completions"
_PAGES_DIR = Path("data/pages/2604.28182v1")


@dataclass
class TestQuery:
    qid: str
    text: str
    page: int
    expected: list[str]


# Directly quoting q37/q38/q39 from data/golden/v3.yaml — figure-grounded
# queries where text-only retrieval scored weakly in ADR 0007's per-subset cut.
QUERIES = [
    TestQuery(
        qid="q37_fig1",
        text=(
            "What two RL training outcomes does the exploration hacking paper's "
            "Figure 1 contrast for locked model organisms?"
        ),
        page=2,
        expected=[
            "Successful RL elicitation (model recovers pre-locking performance)",
            "Successful RL resistance (model stays at locked baseline)",
        ],
    ),
    TestQuery(
        qid="q38_fig2_categories",
        text=(
            "What are the two main categories of exploration hacking strategies "
            "in this paper's taxonomy (Figure 2)?"
        ),
        page=5,
        expected=[
            "Complete under-exploration — reward does not meaningfully increase",
            "Partial under-exploration — splits into Instrumental and Terminal",
        ],
    ),
    TestQuery(
        qid="q39_fig2_subtypes",
        text=(
            "In Figure 2 of this paper, what is the difference between Instrumental "
            "and Terminal subtypes of partial under-exploration?"
        ),
        page=5,
        expected=[
            "Instrumental: reward converges below benign baseline",
            "Terminal: reward may match benign-baseline but with model-preferred behavior",
        ],
    ),
]

MODELS = [
    "qwen/qwen3-vl-8b-instruct",  # cheapest tier
    "qwen/qwen3-vl-32b-instruct",  # likely sweet spot
    "qwen/qwen3-vl-235b-a22b-instruct",  # flagship MoE
    "openai/gpt-4o-mini",  # baseline (already used in Phase 0 smoke eval)
]


def encode_image(path: Path) -> str:
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def ask(api_key: str, model: str, query: str, image_data_url: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    r = httpx.post(_URL, json=payload, headers=headers, timeout=60)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "tokens_in": data.get("usage", {}).get("prompt_tokens", 0),
        "tokens_out": data.get("usage", {}).get("completion_tokens", 0),
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    api_key = os.environ.get("RAG_OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("RAG_OPENROUTER_API_KEY not set; check .env")

    for q in QUERIES:
        page_path = _PAGES_DIR / f"2604.28182v1_p{q.page}.png"
        if not page_path.exists():
            print(f"[skip] {q.qid}: page not rendered yet ({page_path})")
            continue
        print(f"\n========== {q.qid} (page {q.page}) ==========")
        print(f"Q: {q.text}")
        print("Expected facts:")
        for e in q.expected:
            print(f"  - {e}")
        image_data_url = encode_image(page_path)
        for model in MODELS:
            try:
                result = ask(api_key, model, q.text, image_data_url)
            except httpx.HTTPStatusError as exc:
                print(
                    f"\n[{model}] FAILED status={exc.response.status_code} "
                    f"body={exc.response.text[:200]}"
                )
                continue
            except Exception as exc:
                print(f"\n[{model}] FAILED {type(exc).__name__}: {exc}")
                continue
            print(
                f"\n[{model}] {result['elapsed_ms']} ms, "
                f"in={result['tokens_in']}, out={result['tokens_out']}"
            )
            print(result["text"].strip())


if __name__ == "__main__":
    main()
