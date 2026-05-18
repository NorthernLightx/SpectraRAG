"""Rerank-stage latency + correctness bench for the reranker-swap DoE.

No committed latency harness existed, so the DoE's primary response variable
had no instrument; this is it. It times the cross-encoder cost centre directly
-- `CrossEncoder.predict` over `--input-size` (query, doc) pairs, the same call
`BgeReranker.rerank` makes (src/rag/rerank.py:106) minus the negligible
sort / length-norm wrapper -- so timings are faithful to the production rerank
stage. `--input-size` defaults to 50 to match `eval_run --rerank-input-size`
and `--doc-chars` to 1200, the target text-chunk length rerank.py is tuned for.

Run (one model -- the ~5.5s premise gate):
  .venv\\Scripts\\python.exe -m scripts.bench_rerank \\
      --model BAAI/bge-reranker-v2-m3 --iters 20
"""

from __future__ import annotations

import argparse
import statistics
import time

_DOC_FILLER = (
    "Transformer architectures rely on multi-head self-attention to model "
    "long-range dependencies across the input sequence, and recent retrieval "
    "augmented generation systems combine a dense bi-encoder with a sparse "
    "lexical signal before a cross-encoder reranks the fused candidate pool. "
)


def _doc(n_chars: int) -> str:
    base = _DOC_FILLER
    while len(base) < n_chars:
        base += _DOC_FILLER
    return base[:n_chars]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True, help="Repeatable.")
    p.add_argument("--input-size", type=int, default=50)
    p.add_argument("--doc-chars", type=int, default=1200)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()} device={device}")

    query = "How does the proposed method improve retrieval accuracy?"
    doc = _doc(args.doc_chars)
    pairs = [[query, doc] for _ in range(args.input_size)]
    rel = [query, "The proposed method improves retrieval accuracy by reranking "
                   "the fused candidate pool with a cross-encoder before generation."]
    irr = [query, "Medieval European crop rotation alternated cereals with legumes "
                   "to restore soil nitrogen across a three-field system."]

    for model_name in args.model:
        print(f"\n=== {model_name} ===")
        t0 = time.perf_counter()
        try:
            ce = CrossEncoder(model_name, device=device)
        except Exception as e:  # noqa: BLE001 - bench reports load failure, continues
            print(f"LOAD FAILED: {type(e).__name__}: {e}")
            continue
        cold_s = time.perf_counter() - t0

        s_rel, s_irr = (float(x) for x in ce.predict([rel, irr]))
        correctness = "ok" if s_rel > s_irr else "FAIL"

        for _ in range(args.warmup):
            ce.predict(pairs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(args.iters):
            t = time.perf_counter()
            ce.predict(pairs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t) * 1000.0)

        times_ms.sort()
        idx95 = min(len(times_ms) - 1, int(round(0.95 * len(times_ms))) - 1)
        print(
            f"cold_load={cold_s:.1f}s  rel={s_rel:.3f} irr={s_irr:.3f} ({correctness})  "
            f"n={args.iters} input_size={args.input_size} doc_chars={args.doc_chars}\n"
            f"latency_ms p50={statistics.median(times_ms):.1f} "
            f"p95={times_ms[idx95]:.1f} mean={statistics.mean(times_ms):.1f} "
            f"min={times_ms[0]:.1f}"
        )


if __name__ == "__main__":
    main()
