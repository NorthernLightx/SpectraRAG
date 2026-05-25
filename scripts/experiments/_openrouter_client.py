"""Shared OpenRouter wiring for the MMLongBench QA harness (gen + score).

run_mmlb_qa.py and score_mmlb_qa.py both grew an `openrouter` provider path so
the QA protocol can run on a cloud model that local Ollama's daily quota can't
sustain (qwen3-vl:235b-cloud is capped at ~33 calls/day; a 149-q gen + ~182-call
extraction run blows past that). This module concentrates the two pieces both
need:

  1. KEY LOADING. For the FastAPI app the key arrives via pydantic-settings, but
     a plain `python -m scripts.experiments....` invocation does NOT auto-load
     .env into os.environ. So we read os.environ first (the convention in
     scripts/smoke_eval.py: RAG_OPENROUTER_API_KEY, then OPENROUTER_API_KEY) and
     fall back to a minimal .env parse. The parser is dependency-free on purpose
     (python-dotenv is installed but undeclared in pyproject deps; the rest of
     src/llm keeps a tight dependency surface, so we don't lean on it here).

  2. RETRY HEADROOM. OpenRouterClient already retries transport errors + HTTP
     429 with exponential backoff (2s->60s, 6 attempts). Free-tier models 429
     CONSTANTLY ("...is temporarily rate-limited upstream. Please retry
     shortly...", observed on deepseek-v4-flash:free on call 1), so for a long
     batch we widen the cap to give a minute-bounded rate window room to clear.
     The resumable cache in each caller is the real safety net: a query that
     still fails after retries is left uncached and a rerun retries only it.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.llm.openrouter import OpenRouterClient

# Order mirrors scripts/smoke_eval.py: project-prefixed var first, then the bare
# OpenRouter name some tooling exports.
_KEY_ENV_VARS = ("RAG_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")


def _load_key_from_dotenv(env_path: Path) -> str | None:
    """Minimal KEY=VALUE .env reader (no python-dotenv dep). Returns the first
    matching OpenRouter key, stripping surrounding quotes; ignores comments and
    blank lines. Returns None if the file is absent or holds no key."""
    if not env_path.exists():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in _KEY_ENV_VARS:
            return value.strip().strip('"').strip("'")
    return None


def resolve_openrouter_key(env_path: Path = Path(".env")) -> str:
    """os.environ (RAG_OPENROUTER_API_KEY, then OPENROUTER_API_KEY) first, then a
    .env fallback. Raises SystemExit with an actionable message if neither has a
    key, so a missing key fails the script up front rather than on call 1."""
    for var in _KEY_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    from_file = _load_key_from_dotenv(env_path)
    if from_file:
        return from_file
    raise SystemExit(
        f"No OpenRouter key found. Set one of {_KEY_ENV_VARS} in the environment "
        f"or in {env_path} (KEY=sk-or-v1-...)."
    )


def build_openrouter_client(*, timeout: float, env_path: Path = Path(".env")) -> OpenRouterClient:
    """OpenRouterClient with the resolved key and a per-request timeout.

    Free-tier 429s are minute-bounded; OpenRouterClient's own backoff caps at
    60s over 6 attempts. A larger per-request `timeout` matters because vision
    generation over several page PNGs can be slow upstream.
    """
    return OpenRouterClient(api_key=resolve_openrouter_key(env_path), timeout=timeout)
