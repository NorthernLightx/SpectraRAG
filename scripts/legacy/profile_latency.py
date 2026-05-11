"""Profile per-stage retrieval latency on the local dev stack.

Times BM25, dense (Qdrant), RRF fusion, and the Ollama embed roundtrip
over a small set of representative queries. Reports median + p95 per
stage so the breakdown reads as `where does the time actually go`.

Reranker (BGE-rerank-v2-m3) and generation (LLM call) latencies are NOT
measured here — the reranker needs the GPU/CPU model load (~5 s) and
generation hits a paid LLM API. Their typical latencies are documented
inline in `docs/results.md` from the v2 baseline measurements.

Usage:

    .venv/Scripts/python.exe -m scripts.profile_latency

Prerequisites:
- `docker compose up -d qdrant ollama` running (the dev stack default)
- `bootstrap_corpus.py` already populated `RAG_CORPUS_COLLECTION` against
  the same Qdrant
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from src.config.settings import load_settings
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.rag.bm25 import Bm25Index
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.vectorstore import QdrantVectorStore

# Six representative queries — mix of factual, definitional, methodological.
# Kept short so the embedding cost reflects "real query" shape, not pathological
# long-context inputs.
SAMPLE_QUERIES = [
    "what is the inter-basin gain criterion",
    "explain the architecture of the proposed method",
    "what evaluation metrics are used in the study",
    "what is the main contribution of this paper",
    "describe the experimental setup and dataset size",
    "what are the limitations and failure modes of the approach",
]


@dataclass(frozen=True)
class StageStats:
    name: str
    median_ms: float
    p95_ms: float
    samples: int


def _stats(name: str, values: list[float]) -> StageStats:
    if len(values) < 2:
        return StageStats(name=name, median_ms=values[0], p95_ms=values[0], samples=len(values))
    sorted_vals = sorted(values)
    median = statistics.median(sorted_vals)
    # Approximate p95 — for n=6 samples this is just the second-largest value.
    idx = max(0, round(0.95 * len(sorted_vals)) - 1)
    p95 = sorted_vals[idx]
    return StageStats(name=name, median_ms=median, p95_ms=p95, samples=len(values))


async def main() -> None:
    settings = load_settings()
    embedder = OllamaBgeEmbedder(base_url=settings.ollama_base_url)
    vectorstore = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.corpus_collection,
        dim=embedder.dim,
    )

    print(f"Loading corpus from {settings.qdrant_url} ({settings.corpus_collection})...")
    chunks = await vectorstore.scroll_chunks()
    bm25: Bm25Index | None = None
    if chunks:
        print(f"Loaded {len(chunks)} chunks; building BM25 index...")
        bm25 = Bm25Index()
        bm25.add(chunks)
    else:
        # Older collections were written before the full-Chunk payload schema
        # (`{chunk_id, paper_id}` only). We can still time the embed + dense
        # path; BM25 + RRF are skipped with a note.
        print(
            f"Collection {settings.corpus_collection!r} uses the legacy payload "
            "schema (chunk_id only). Profiling embed + dense only; BM25/RRF "
            "skipped — re-ingest with `bootstrap_corpus.py --force` for the "
            "full profile."
        )

    embed_ms: list[float] = []
    dense_ms: list[float] = []
    bm25_ms: list[float] = []
    rrf_ms: list[float] = []

    print("Warming up embedder...")
    [_warmup] = await embedder.embed_texts([SAMPLE_QUERIES[0]])

    print(f"Profiling {len(SAMPLE_QUERIES)} queries...")
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        # Stage 1: embed
        t0 = time.perf_counter()
        [vector] = await embedder.embed_texts([query])
        embed_ms.append((time.perf_counter() - t0) * 1000)

        # Stage 2: dense (Qdrant)
        t0 = time.perf_counter()
        dense_hits = await vectorstore.search(vector, top_k=50)
        dense_ms.append((time.perf_counter() - t0) * 1000)

        if bm25 is not None:
            # Stage 3: BM25 (in-process)
            t0 = time.perf_counter()
            sparse_hits = bm25.search(query, top_k=50)
            bm25_ms.append((time.perf_counter() - t0) * 1000)

            # Stage 4: RRF fusion (in-process)
            dense_ranked = [RankedItem(id=h.chunk_id, score=h.score) for h in dense_hits]
            sparse_ranked = [RankedItem(id=h.chunk_id, score=h.score) for h in sparse_hits]
            t0 = time.perf_counter()
            _ = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_k=10)
            rrf_ms.append((time.perf_counter() - t0) * 1000)

            print(
                f"  [{i}/{len(SAMPLE_QUERIES)}] {query[:50]:<50} "
                f"embed {embed_ms[-1]:.0f} dense {dense_ms[-1]:.0f} "
                f"bm25 {bm25_ms[-1]:.0f} rrf {rrf_ms[-1]:.0f}"
            )
        else:
            print(
                f"  [{i}/{len(SAMPLE_QUERIES)}] {query[:50]:<50} "
                f"embed {embed_ms[-1]:.0f} dense {dense_ms[-1]:.0f}"
            )

    stages = [
        _stats("Embed query (Ollama bge-m3, CPU)", embed_ms),
        _stats("Dense search (Qdrant top-50)", dense_ms),
    ]
    if bm25 is not None:
        stages.append(_stats("BM25 search (in-process top-50)", bm25_ms))
        stages.append(_stats("RRF fusion (top-10)", rrf_ms))

    print()
    print(f"{'Stage':<40} {'median (ms)':>12} {'p95 (ms)':>10}")
    print("-" * 65)
    for s in stages:
        print(f"{s.name:<40} {s.median_ms:>12.1f} {s.p95_ms:>10.1f}")

    sum_median = sum(s.median_ms for s in stages)
    print(f"{'sum (retrieval-only)':<40} {sum_median:>12.1f}")

    print()
    print("Reranker + generation are NOT measured here.")
    print("Typical from v2 baseline (golden v2):")
    print("  Reranker (BGE-rerank-v2-m3, GPU)      ~5500 ms")
    print("  Generation (qwen2.5:7b, local LLM)    ~60000 ms (Ollama, GPU)")
    print("  Generation (gpt-4o-mini, OpenRouter)  ~2000 ms (network + remote)")


if __name__ == "__main__":
    asyncio.run(main())
