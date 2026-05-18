"""Unattended reranker-swap DoE driver.

Runs the keyless deterministic core only: a latency bench plus a retrieval-only
paired sweep (incumbent vs candidate cross-encoders) on golden v3 and
MMLongBench, then a paired-bootstrap analysis. No LLM keys, no router, no
ColQwen2 -- this isolates the reranker (the only variable) and avoids the
8 GB-card OOM that an unattended ColQwen2 load risks. Judged metrics
(context_precision etc.) are intentionally excluded: unbounded local-LLM
latency is not safe to run blind; that is an attended Ollama follow-up.

Retrieval metrics are deterministic (eval_run run_id is a content hash), so one
run per (model, golden) is exact and pairing is by query position.

Output: data/eval/runs/doe-<ts>/ -- per-run JSONs, bench.txt, REPORT.md, driver.log
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable  # the venv python this driver was launched with
RUNS = ROOT / "data" / "eval" / "runs"
PDFS = sorted(str(p) for p in (ROOT / "data" / "papers").glob("*.pdf"))
GOLDENS = [("v3", "data/golden/v3.yaml"), ("mmlb", "data/golden/mmlongbench-v1.yaml")]
CONTROL = "BAAI/bge-reranker-v2-m3"
CANDIDATES = ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"]
MODELS = [CONTROL, *CANDIDATES]
RETRIEVAL_FIELDS = ("ndcg_at_5", "recall_at_10", "mrr")

OUT = RUNS / f"doe-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    log("RUN " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    try:
        r = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return 124, ""
    tail = "\n".join((r.stdout or "").splitlines()[-8:])
    log(f"  rc={r.returncode}\n{tail}")
    if r.returncode != 0:
        log("  STDERR " + "\n".join((r.stderr or "").splitlines()[-8:]))
    return r.returncode, r.stdout or ""


def newest_json(before: set[Path]) -> Path | None:
    after = set(OUT.glob("run-*.json"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    return new[-1] if new else None


def per_query(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("per_query", [])


def _val(q: dict, field: str) -> float | None:
    for c in (q.get("retrieval") or {}, q.get("generation") or {}):
        if c.get(field) is not None:
            return float(c[field])
    return None


def bootstrap(deltas: list[float], n: int = 5000) -> tuple[float, float, float]:
    rnd = random.Random(0)
    m = len(deltas)
    means = sorted(
        sum(deltas[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n)
    )
    return sum(deltas) / m, means[int(0.025 * n)], means[int(0.975 * n)]


def macro(pq: list[dict], field: str) -> float | None:
    vals = [
        v
        for q in pq
        if q.get("category") != "out_of_corpus"
        for v in (_val(q, field),)
        if v is not None
    ]
    return sum(vals) / len(vals) if vals else None


def analyse(runs: dict[tuple[str, str], Path]) -> str:
    out = ["# Reranker-swap DoE — deterministic retrieval results\n"]
    out.append(
        "Premise note: the ~5.5 s reranker figure (results.md:39) is a stale "
        "v2-baseline/legacy-profiler number. Measured faithfully on this GPU the "
        "incumbent rerank stage is ~1.0 s p50 (see bench.txt). Latency is a minor "
        "lever; this sweep tests whether a cheaper reranker holds ranking quality.\n"
    )
    for gname, _ in GOLDENS:
        cpath = runs.get((CONTROL, gname))
        if not cpath:
            out.append(f"\n## {gname}: control run MISSING — skipped\n")
            continue
        cpq = per_query(cpath)
        out.append(f"\n## {gname} (n_total={len(cpq)})\n")
        out.append("| model | field | macro | Δ vs control | 95% CI | better/worse/eq |")
        out.append("|---|---|---|---|---|---|")
        for model in MODELS:
            rp = runs.get((model, gname))
            if not rp:
                out.append(f"| {model} | — | RUN MISSING | | | |")
                continue
            mpq = per_query(rp)
            for f in RETRIEVAL_FIELDS:
                mm = macro(mpq, f)
                if model == CONTROL:
                    out.append(f"| {model} | {f} | {mm:.4f} | — | — | — |")
                    continue
                pairs = [
                    (_val(cpq[i], f), _val(mpq[i], f))
                    for i in range(min(len(cpq), len(mpq)))
                    if cpq[i].get("category") != "out_of_corpus"
                ]
                d = [b - a for a, b in pairs if a is not None and b is not None]
                if not d:
                    out.append(f"| {model} | {f} | {mm} | no pairs | | |")
                    continue
                mean, lo, hi = bootstrap(d)
                bw = sum(x > 1e-9 for x in d)
                ww = sum(x < -1e-9 for x in d)
                out.append(
                    f"| {model} | {f} | {mm:.4f} | {mean:+.4f} | "
                    f"[{lo:+.4f}, {hi:+.4f}] | {bw}/{ww}/{len(d) - bw - ww} |"
                )
    out.append("\n## Per-category mean Δ (candidate − control), ndcg_at_5\n")
    out.append("| golden | model | category | mean Δ | n |")
    out.append("|---|---|---|---|---|")
    for gname, _ in GOLDENS:
        cpath = runs.get((CONTROL, gname))
        if not cpath:
            continue
        cpq = per_query(cpath)
        for model in CANDIDATES:
            rp = runs.get((model, gname))
            if not rp:
                continue
            mpq = per_query(rp)
            cats: dict[str, list[float]] = {}
            for i in range(min(len(cpq), len(mpq))):
                a, b = _val(cpq[i], "ndcg_at_5"), _val(mpq[i], "ndcg_at_5")
                if a is None or b is None or cpq[i].get("category") == "out_of_corpus":
                    continue
                cats.setdefault(cpq[i].get("category", "?"), []).append(b - a)
            for cat, xs in sorted(cats.items()):
                out.append(
                    f"| {gname} | {model} | {cat} | "
                    f"{sum(xs) / len(xs):+.4f} | {len(xs)} |"
                )
    out.append("\n## Latency bench\n```\n" + (OUT / "bench.txt").read_text() + "\n```\n")
    return "\n".join(out)


def main() -> None:
    try:  # Windows console is cp1252; the report contains U+0394 (Δ)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"DoE start. {len(PDFS)} PDFs. out={OUT}")

    # Keyless precheck: one PDF, smoke golden, retrieval-only. If the stack
    # (Qdrant / embedder / ingest / rerank) is down, fail the night fast.
    if not PDFS:
        log("ABORT: no PDFs in data/papers")
        return
    before = set(OUT.glob("run-*.json"))
    rc, _ = run(
        [PY, "-m", "scripts.eval_run", "--pdf", PDFS[0],
         "--golden", "data/golden/v1.yaml", "--rerank",
         "--rerank-model", CONTROL, "--rerank-device", "cuda",
         "--postgres-dsn", "", "--output-dir", str(OUT),
         "--collection", "doe_precheck"],
        timeout=1800,
    )
    if rc != 0 or newest_json(before) is None:
        log("ABORT: precheck failed — stack not healthy; not burning the night.")
        return
    log("Precheck OK.")

    # Latency bench (incumbent + candidates).
    rc, sout = run(
        [PY, "-m", "scripts.bench_rerank", "--iters", "20",
         *sum([["--model", m] for m in MODELS], [])],
        timeout=3600,
    )
    (OUT / "bench.txt").write_text(sout, encoding="utf-8")

    # Deterministic retrieval-only sweep.
    runs: dict[tuple[str, str], Path] = {}
    for model in MODELS:
        slug = model.replace("/", "_").replace("-", "_").replace(".", "_")
        for gname, gpath in GOLDENS:
            before = set(OUT.glob("run-*.json"))
            rc, _ = run(
                [PY, "-m", "scripts.eval_run", "--pdf", *PDFS,
                 "--golden", gpath, "--rerank", "--rerank-model", model,
                 "--rerank-device", "cuda", "--postgres-dsn", "",
                 "--output-dir", str(OUT),
                 "--collection", f"doe_{slug}_{gname}"],
                timeout=10800,
            )
            j = newest_json(before)
            if rc == 0 and j is not None:
                runs[(model, gname)] = j
                log(f"  saved {model}/{gname} -> {j.name}")
            else:
                log(f"  FAILED {model}/{gname} (rc={rc}) — continuing")

    report = analyse(runs)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    log("DoE done. REPORT.md written.")
    print("\n===REPORT===\n" + report)


if __name__ == "__main__":
    main()
