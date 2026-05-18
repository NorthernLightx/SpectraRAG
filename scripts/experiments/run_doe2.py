"""Phase 2 of the reranker-swap DoE: judged metrics via local Ollama.

Phase 1 (scripts/experiments/run_doe.py) covers the keyless deterministic sweep.
This adds what the keyless half cannot see: the LLM-judged metrics
(faithfulness, answer_relevance, context_precision -- the weak ~0.296 one) on
golden v3, plus a judge-variance noise floor so the cross-model deltas are
interpretable rather than point guesses.

Constraints baked in:
- gemma3:4b for BOTH generator and judge -- one model avoids Ollama
  model-swap thrash on the 8 GB card and stays co-resident with the bge-m3
  embedder. Generator/judge already default to the ollama provider in
  eval_run; the default tag (qwen2.5:7b) is NOT installed, so it is pinned.
- No router / no ColQwen2: isolates the reranker and avoids the 8 GB OOM.
- Hard wall-clock budget: stop launching new runs past BUDGET_H and analyse
  whatever finished. An unattended run must not surprise the morning.

Run-plan order puts the decision-critical comparison first (incumbent then
the two candidates) so a truncated budget still yields the cross-model
contrast; incumbent reps 2-3 are the noise floor and degrade gracefully.

Output: data/eval/runs/doe2-<ts>/  -- per-run JSONs, REPORT2.md, driver.log
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PDFS = sorted(str(p) for p in (ROOT / "data" / "papers").glob("*.pdf"))
GOLDEN = "data/golden/v3.yaml"
LLM = "gemma3:4b"
BUDGET_H = 5.0
PER_RUN_TIMEOUT = 7200  # 2 h; budget guard (not this) bounds the whole run

CONTROL = "BAAI/bge-reranker-v2-m3"
CANDIDATES = ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"]
# bge-reranker-base (CANDIDATES[0]) dropped after Phase 1: v3 ndcg@5
# Δ -0.1383, bootstrap CI [-0.279, -0.009] excludes 0 -- decisively
# regressive, so re-confirming it via judged metrics wastes budget.
# Phase 2 focuses on the live question: does ms-marco-MiniLM-L-6-v2's
# retrieval non-inferiority carry through to answer quality.
# (model, tag). Decision-critical first; incumbent rep2/rep3 are the noise floor.
PLAN = [
    (CONTROL, "rep1"),
    (CANDIDATES[1], "r1"),
    (CONTROL, "rep2"),
    (CONTROL, "rep3"),
]
GEN_FIELDS = ("faithfulness", "answer_relevance", "context_precision", "citation_grounding")
RET_FIELDS = ("ndcg_at_5", "recall_at_10", "mrr")

OUT = ROOT / "data" / "eval" / "runs" / f"doe2-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str]) -> int:
    log("RUN " + " ".join(cmd[:7]) + " ...")
    try:
        r = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=PER_RUN_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {PER_RUN_TIMEOUT}s")
        return 124
    log(f"  rc={r.returncode}\n" + "\n".join((r.stdout or "").splitlines()[-6:]))
    if r.returncode != 0:
        log("  STDERR " + "\n".join((r.stderr or "").splitlines()[-6:]))
    return r.returncode


def newest_json(before: set[Path]) -> Path | None:
    new = sorted(set(OUT.glob("run-*.json")) - before, key=lambda p: p.stat().st_mtime)
    return new[-1] if new else None


def per_query(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("per_query", [])


def val(q: dict, field: str) -> float | None:
    for c in (q.get("retrieval") or {}, q.get("generation") or {}):
        if c.get(field) is not None:
            return float(c[field])
    return None


def macro(pq: list[dict], field: str) -> float | None:
    # retrieval fields: in-corpus only; generation fields: all with a value
    # (mirrors scripts/check_regression.py _macro_mean).
    qs = [q for q in pq if q.get("category") != "out_of_corpus"] if field in RET_FIELDS else pq
    vs = [v for q in qs if (v := val(q, field)) is not None]
    return sum(vs) / len(vs) if vs else None


def bootstrap(d: list[float], n: int = 5000) -> tuple[float, float, float]:
    rnd = random.Random(0)
    m = len(d)
    ms = sorted(sum(d[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(d) / m, ms[int(0.025 * n)], ms[int(0.975 * n)]


def analyse(runs: dict[tuple[str, str], Path]) -> str:
    o = ["# Reranker-swap DoE Phase 2 — judged metrics (gemma3:4b, golden v3)\n"]
    base = runs.get((CONTROL, "rep1"))
    if not base:
        return "\n".join(o + ["\nincumbent rep1 MISSING — no comparison possible.\n"])
    bpq = per_query(base)

    o.append("## Incumbent noise floor (v3-judged reps)\n")
    o.append("| metric | rep1 | rep2 | rep3 | range |")
    o.append("|---|---|---|---|---|")
    reps = [runs.get((CONTROL, r)) for r in ("rep1", "rep2", "rep3")]
    for f in GEN_FIELDS:
        vs = [macro(per_query(r), f) if r else None for r in reps]
        sv = [x for x in vs if x is not None]
        rng = f"{max(sv) - min(sv):.4f}" if len(sv) > 1 else "n/a"
        o.append(
            "| " + f + " | " + " | ".join(f"{x:.4f}" if x is not None else "—" for x in vs)
            + f" | {rng} |"
        )

    o.append("\n## Candidate vs incumbent (paired, v3)\n")
    o.append("| model | metric | macro | Δ vs incumbent | 95% CI | better/worse/eq |")
    o.append("|---|---|---|---|---|---|")
    for model in (CONTROL, CANDIDATES[1]):  # bge-base dropped post-Phase-1
        rp = base if model == CONTROL else runs.get((model, "r1"))
        if not rp:
            o.append(f"| {model} | — | RUN MISSING | | | |")
            continue
        mpq = per_query(rp)
        for f in (*RET_FIELDS, *GEN_FIELDS):
            mm = macro(mpq, f)
            if mm is None:
                continue
            if model == CONTROL:
                o.append(f"| {model} | {f} | {mm:.4f} | — | — | — |")
                continue
            incorp = f in RET_FIELDS
            pairs = [
                (val(bpq[i], f), val(mpq[i], f))
                for i in range(min(len(bpq), len(mpq)))
                if not (incorp and bpq[i].get("category") == "out_of_corpus")
            ]
            d = [b - a for a, b in pairs if a is not None and b is not None]
            if not d:
                o.append(f"| {model} | {f} | {mm:.4f} | no pairs | | |")
                continue
            mean, lo, hi = bootstrap(d)
            bw = sum(x > 1e-9 for x in d)
            ww = sum(x < -1e-9 for x in d)
            o.append(
                f"| {model} | {f} | {mm:.4f} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] "
                f"| {bw}/{ww}/{len(d) - bw - ww} |"
            )
    o.append(
        "\n_Read deltas against the incumbent noise-floor range above: a Δ "
        "smaller than the range is judge noise, not signal._\n"
    )
    return "\n".join(o)


def main() -> None:
    try:  # Windows console is cp1252; the report contains U+0394 (Δ)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log(f"Phase 2 start. {len(PDFS)} PDFs. budget={BUDGET_H}h out={OUT}")
    if not PDFS:
        log("ABORT: no PDFs")
        return

    runs: dict[tuple[str, str], Path] = {}
    for model, tag in PLAN:
        if (time.time() - start) / 3600 > BUDGET_H:
            log(f"BUDGET {BUDGET_H}h reached — skipping remaining; analysing partial.")
            break
        slug = model.replace("/", "_").replace("-", "_").replace(".", "_")
        before = set(OUT.glob("run-*.json"))
        rc = run(
            [PY, "-m", "scripts.eval_run", "--pdf", *PDFS, "--golden", GOLDEN,
             "--rerank", "--rerank-model", model, "--rerank-device", "cuda",
             "--generate", "--generator-provider", "ollama", "--generator-model", LLM,
             "--judge", "--judge-provider", "ollama", "--judge-model", LLM,
             "--postgres-dsn", "", "--output-dir", str(OUT),
             "--collection", f"doe2_{slug}_{tag}"]
        )
        j = newest_json(before)
        if rc == 0 and j is not None:
            runs[(model, tag)] = j
            log(f"  saved {model}/{tag} -> {j.name}")
        else:
            log(f"  FAILED {model}/{tag} (rc={rc}) — continuing")

    (OUT / "REPORT2.md").write_text(analyse(runs), encoding="utf-8")
    log("Phase 2 done. REPORT2.md written.")
    print("\n===REPORT2===\n" + (OUT / "REPORT2.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
