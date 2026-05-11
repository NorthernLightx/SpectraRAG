"""Smoke pre-flight: run the full production stack on golden v1 (5q, 1 paper).

The full v3 eval takes 60-90 minutes and burns GPU + OpenRouter budget. When
it fails mid-run (Ollama hangs, GPU contention segfaults, schema drift in a
dependency), the only signal you get is at the end. This script catches
those failure modes in ~5 minutes by exercising the same code path on a
tiny golden set: 1 paper, 5 queries, all the production flags on.

Run before any v3 / mmlongbench eval. If smoke fails, the bigger eval
will also fail — diagnose now, save the GPU time.

Usage:
    .venv/Scripts/python.exe -m scripts.smoke_eval

Or with custom paper / verbosity:
    .venv/Scripts/python.exe -m scripts.smoke_eval \\
        --pdf data/papers/2604.22753v1.pdf --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_SMOKE_PAPER = "data/papers/2604.22753v1.pdf"
_SMOKE_GOLDEN = "data/golden/v1.yaml"
_SMOKE_COLLECTION = "smoke_eval_preflight"
_SMOKE_TIMEOUT_S = 600  # 10 min hard cap; smoke should finish in 3-5 min.


def _check_prereqs() -> list[str]:
    """Return a list of human-readable prereq violations. Empty list = green light."""
    errors: list[str] = []
    if not Path(_SMOKE_PAPER).exists():
        errors.append(f"smoke paper missing: {_SMOKE_PAPER}")
    if not Path(_SMOKE_GOLDEN).exists():
        errors.append(f"golden v1 missing: {_SMOKE_GOLDEN}")
    api_key = os.environ.get("RAG_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        errors.append(
            "RAG_OPENROUTER_API_KEY (or OPENROUTER_API_KEY) not set — generate/judge will fail."
        )
    return errors


def _build_eval_command(*, pdf: Path) -> list[str]:
    """Production stack on a tiny golden. All Tier 1 features on. Args are
    a fixed list — no user-supplied shell strings, so no injection surface."""
    return [
        sys.executable,
        "-m",
        "scripts.eval_run",
        "--pdf",
        str(pdf),
        "--golden",
        _SMOKE_GOLDEN,
        # Production stack — match committed baseline f844619927e0.
        "--rerank",
        "--rerank-length-norm",
        "--router",
        "--paper-id-filter",
        "--region-number-boost",
        "--refusal-score-threshold",
        "0.105",
        # Generation + judge so the smoke also exercises OpenRouter.
        "--generate",
        "--generator-provider",
        "openrouter",
        "--generator-model",
        "openai/gpt-4o-mini",
        "--judge",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "openai/gpt-4o-mini",
        # Skip Postgres persistence — smoke shouldn't touch shared infra.
        "--postgres-dsn",
        "",
        # Dedicated collection so smoke doesn't pollute the real eval state.
        "--collection",
        _SMOKE_COLLECTION,
        "--output-dir",
        "data/eval/runs",
    ]


async def _run_smoke(cmd: list[str]) -> int:
    """Run the smoke eval as a child process via execvp (asyncio's safe path);
    stream merged stdout/stderr to this process. No shell, no injection."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.STDOUT,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=_SMOKE_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        print(
            f"\n[smoke_eval] FAILED: hard timeout at {_SMOKE_TIMEOUT_S}s. The full eval would "
            "almost certainly hang or fail too — investigate before launching v3."
        )
        return 124
    return proc.returncode or 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(_SMOKE_PAPER),
        help="Override the smoke paper. Default is the canonical v1 paper.",
    )
    args = parser.parse_args()

    errors = _check_prereqs()
    if errors:
        print("[smoke_eval] prerequisite failures:")
        for e in errors:
            print(f"  - {e}")
        return 2

    cmd = _build_eval_command(pdf=args.pdf)
    print(f"[smoke_eval] running: {' '.join(cmd)}")
    started = time.monotonic()
    code = asyncio.run(_run_smoke(cmd))
    elapsed = time.monotonic() - started

    if code == 0:
        print(f"\n[smoke_eval] OK in {elapsed:.1f}s — full v3 eval is green-lit.")
    else:
        print(
            f"\n[smoke_eval] FAILED with exit code {code} after {elapsed:.1f}s. "
            "Investigate before launching v3 — the same failure mode will reproduce."
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
