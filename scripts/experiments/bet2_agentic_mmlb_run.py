"""Bet 2 decisive test, retrieval arm: agentic decomposition on MMLongBench-Doc.

ADR 0019 measured the agentic tier (decompose -> retrieve-per-subquery -> RRF)
on golden v3 (text-heavy arXiv, n=39): figure improved, factual/table
regressed, net within the gemma3:4b judge noise band. bet2_selective_gate.py
re-scored the category-gate idea on v3 and found every per-category delta inside
that noise band -- INCONCLUSIVE at v3 sizes. The decisive test the strategist
named: a fresh agentic run on MMLongBench (larger n, page-level gold), scored at
PAGE granularity against the same depth-50 baseline.

This driver runs ONLY the text leg wrapped in AgenticRetriever, over the SAME
pre-ingested `routing_study` collection the committed depth-50 baseline used
(data/eval/runs/depth50-20260525-015216/depth50.json), so the A/B is
apples-to-apples against that run's `text_top50` leg. The agentic tier is
text-side, so this is a clean text-vs-text comparison: no ColQwen2, no visual
leg, no GPU contention (bge-m3 embedder + gemma3:4b decomposer are both light).

  Per query (top_k=DEPTH):
    gemma3:4b decomposes -> N atomic sub-questions
    each sub-question -> the SAME PipelineRetriever (dense+BM25+RRF+rerank
                         length-norm, candidate_pool=DEPTH) the baseline used
    RRF-fuse the per-subquery rankings -> agentic_top50
  A query whose decomposition reduces to [original] is byte-for-byte identical
  to the baseline text leg (AgenticRetriever short-circuits to a single base
  retrieval), which is what makes the divergence sanity check meaningful.

Output: data/eval/runs/bet2-agentic-<ts>/agentic.json  (+ driver.log)
  per_query[i] = {query_id, category, n_subqueries, subqueries, agentic_top50}

Score + gate it with the companion:
    .venv/Scripts/python.exe -m scripts.experiments.bet2_mmlb_gate \
        --agentic data/eval/runs/bet2-agentic-<ts>/agentic.json \
        --baseline data/eval/runs/depth50-20260525-015216/depth50.json \
        --golden data/golden/mmlongbench-v1.yaml

Run on the RTX 3070 box (Qdrant + Ollama up; text-only, no ColQwen2):
    .venv/Scripts/python.exe -m scripts.experiments.bet2_agentic_mmlb_run
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from scripts.rescore_mmlb_pages import rescore
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.llm.ollama_chat import OllamaChatClient
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.agentic import AgenticRetriever
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Query

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN = ROOT / "data" / "golden" / "mmlongbench-v1.yaml"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"

# Local decomposer. Cloud Ollama weekly quota is exhausted this session, and
# gemma3:4b is the shipped router classifier model (Settings.classifier_ollama_model),
# so it is the honest local choice for query splitting too.
DECOMPOSE_MODEL = "gemma3:4b"

# Same collection that backs the committed depth-50 baseline (driver.log:
# "loaded 2361 chunks from 'routing_study'"). Re-ingesting in-process exhausts
# docling/native memory on the big PDFs; the chunks already exist, so scroll
# them back and rebuild BM25 + chunks_by_id in process.
COLLECTION = "routing_study"

# Depth to retrieve / persist. 50 matches the baseline's text_top50 leg so the
# A/B compares the same rank depth. Also the candidate-pool / rerank-input
# ceiling on the text leg.
DEPTH = 50

# Cap on sub-questions per query (the ADR 0019 / eval_run.py default).
MAX_SUBQUERIES = 4

OUT = ROOT / "data" / "eval" / "runs" / f"bet2-agentic-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _recall10_macro(rescored: dict[str, Any], subset: set[str] | None) -> tuple[int, float]:
    """Macro recall@10 over in-corpus scorable queries (optionally one category).
    Mirrors diagnose_depth50_run._recall10_macro so the sanity gate uses the
    same subset the baseline self-check reported."""
    rows = [
        pq
        for pq in rescored["per_query"]
        if pq.get("category") != "out_of_corpus"
        and (pq.get("retrieval") or {}).get("recall_at_10") is not None
        and (subset is None or pq.get("category") in subset)
    ]
    if not rows:
        return 0, 0.0
    return len(rows), sum(float(r["retrieval"]["recall_at_10"]) for r in rows) / len(rows)


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"Bet2 agentic-retrieval run start. out={OUT}  DEPTH={DEPTH}  model={DECOMPOSE_MODEL}")

    # --- load the pre-ingested 20-doc corpus from its Qdrant collection, same
    # as the depth-50 baseline. scroll_chunks() rebuilds BM25 + chunks_by_id. ---
    embedder = OllamaBgeEmbedder(base_url=OLLAMA)
    vs = QdrantVectorStore(url=QDRANT, collection_name=COLLECTION, dim=embedder.dim)
    chunks = await vs.scroll_chunks()
    if not chunks:
        log(f"ABORT: collection {COLLECTION!r} empty or missing.")
        return
    bm25 = Bm25Index()
    bm25.add(chunks)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    log(f"loaded {len(chunks)} chunks from {COLLECTION!r}")

    # Identical text leg to the baseline: dense+BM25+RRF, rerank with length-norm,
    # candidate_pool=DEPTH, rerank_input_size=DEPTH.
    text = PipelineRetriever(
        embedder=embedder,
        vectorstore=vs,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        reranker=BgeReranker(length_norm=True),
        candidate_pool=DEPTH,
        rerank_input_size=DEPTH,
    )
    # Agentic wrapper: gemma3:4b decomposes, each sub-question hits `text`, RRF.
    agentic = AgenticRetriever(
        base=text,
        llm=OllamaChatClient(base_url=OLLAMA),
        model=DECOMPOSE_MODEL,
        max_subqueries=MAX_SUBQUERIES,
    )

    gs = load_golden_set(GOLDEN)
    golden_doc = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    log(f"golden: {len(gs.queries)} queries")

    # --- one pass: agentic retrieval at DEPTH for every query. _decompose is
    # called separately to RECORD the sub-questions (retrieve() does not surface
    # them), then retrieve() produces the fused ranking. retrieve() decomposes
    # again internally; the extra call is deterministic (temperature=0) and cheap
    # (~0.3s), and not duplicating the tier's fan-out keeps the A/B running the
    # real shipped AgenticRetriever rather than a reimplementation of it. ---
    per_query: list[dict[str, Any]] = []
    n_decomposed = 0
    examples_logged = 0
    failed: list[str] = []

    def _save() -> None:
        (OUT / "agentic.json").write_text(
            json.dumps(
                {"depth": DEPTH, "model": DECOMPOSE_MODEL, "per_query": per_query}, indent=2
            ),
            encoding="utf-8",
        )

    for i, q in enumerate(gs.queries):
        # The retrieval path is verified working, but under concurrent
        # per-subquery rerank load on the 8GB box a Qdrant HTTP call occasionally
        # times out (ResponseHandlingException). Retry transient failures; skip a
        # query that exhausts all attempts rather than killing the whole pass.
        subqueries: list[str] = []
        res_ids: list[str] = []
        ok = False
        for attempt in range(1, 5):
            try:
                subqueries = await agentic._decompose(q.text)
                res = await agentic.retrieve(Query(text=q.text, top_k=DEPTH))
                res_ids = [r.chunk_id for r in res]
                ok = True
                break
            except Exception as e:  # broad on purpose: any transient transport error is retried
                log(f"  {q.query_id} attempt {attempt}/4 failed: {type(e).__name__}")
                if attempt < 4:
                    await asyncio.sleep(2.0 * attempt)
        if not ok:
            failed.append(q.query_id)
            log(f"  SKIP {q.query_id} after 4 attempts")
            continue
        if len(subqueries) > 1:
            n_decomposed += 1
            if examples_logged < 4:
                log(f"  decomp example {q.query_id} ({q.category}): {q.text!r}")
                for sub in subqueries:
                    log(f"      -> {sub!r}")
                examples_logged += 1
        per_query.append(
            {
                "query_id": q.query_id,
                "category": q.category,
                "n_subqueries": len(subqueries),
                "subqueries": subqueries,
                "agentic_top50": res_ids,
            }
        )
        if (i + 1) % 10 == 0:
            _save()
            log(f"  agentic: {i + 1}/{len(gs.queries)}  (decomposed so far: {n_decomposed})")
    _save()
    if failed:
        log(f"WARNING: {len(failed)} queries failed all retries and were skipped: {failed}")
    log(
        f"agentic run complete. decomposed {n_decomposed}/{len(gs.queries)} queries "
        f"(rest reduced to [original] -> identity to baseline text leg)"
    )
    log(f"wrote {OUT / 'agentic.json'}")

    # --- sanity gate: score agentic TOP-10 page-recall. This is NOT a pass/fail
    # threshold (agentic may legitimately move recall either way) -- it just
    # confirms the run scored on the same n=107 subset the baseline self-check
    # used, so the companion gate's A/B is on the same denominator. ---
    agentic10_pq = [
        {
            "query_id": r["query_id"],
            "category": r["category"],
            "retrieved_chunk_ids": r["agentic_top50"][:10],
        }
        for r in per_query
    ]
    rescored10 = rescore({"per_query": agentic10_pq}, golden_doc)
    n_all, r10_all = _recall10_macro(rescored10, None)
    n_fig, r10_fig = _recall10_macro(rescored10, {"figure"})
    log(
        f"SANITY agentic@10: all n={n_all} recall@10={r10_all:.4f} "
        f"(baseline text-only@10 ref = 0.6184, n=107)"
    )
    log(f"SANITY agentic@10: figure n={n_fig} recall@10={r10_fig:.4f}")
    if n_decomposed == 0:
        log(
            "WARNING: decomposer emitted a single line for EVERY query -- agentic "
            "is byte-for-byte identical to the baseline text leg. The decomposer "
            "silently no-op'd; the A/B would be a null result, not a finding. "
            "Investigate the prompt / model before trusting any gate output."
        )

    print(
        "\nNext: score + gate with\n"
        f"  .venv/Scripts/python.exe -m scripts.experiments.bet2_mmlb_gate \\\n"
        f"      --agentic {OUT / 'agentic.json'} \\\n"
        "      --baseline data/eval/runs/depth50-20260525-015216/depth50.json \\\n"
        f"      --golden {GOLDEN}\n",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
