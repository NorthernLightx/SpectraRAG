"""Depth-50 retrieval run on the MMLongBench corpus — per-leg, for the
figure-miss decomposition.

The committed baselines store only the fused top-10 per query, so
coverage@all == recall@10 trivially and we cannot tell ranking-loss from
true-miss (strategy session 2026-05-24). This driver re-runs the SAME
shipped stack (text = dense+BM25+RRF+rerank with length-norm; visual =
ColQwen2-v1.0; llm-router = gemma3:4b) but at top_k=50, and persists EACH
leg's top-50 page ranking separately plus the fused router output. The
companion `diagnose_figure_misses.py` consumes the output.

Why top_k=50 still reproduces the shipped recall@10 (verified analytically):
RRF score of a page is 1/(k + rank), strictly decreasing in rank, so pages
at per-leg ranks 11-50 can never displace pages at ranks 0-9 in the fused
head. The fused top-10 is therefore identical whether each leg feeds in at
depth 10 or 50 — top_k just truncates later. This run observes the same
system deeper; it does not change it. (Sanity-checked at the end: the
fused-top-10 recall here must match the committed router baseline.)

Structure / corpus / sanity-gate are lifted from
`scripts/experiments/study_routing.py` — same 20 docs, same ingest, same
gate. The ONLY new behaviour: run text and visual legs directly (not via the
opaque RoutingRetriever.retrieve) so each leg's pre-fusion ranking is
observable, and fuse with the real `_fuse_page_level` for the router output.
No `src/rag/` change — the legs are already separable objects here.

Output: data/eval/runs/depth50-<ts>/depth50.json  (+ driver.log)
  per_query[i] = {query_id, category, text_top50, visual_top50, fused_top50}

Run on the RTX 3070 box (ColQwen2 + Ollama up; ~15 min):
    .venv/Scripts/python.exe -m scripts.experiments.diagnose_depth50_run
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
from src.ingestion.visual import render_pages
from src.llm.ollama_chat import OllamaChatClient
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.classifier_llm import LLMQueryClassifier
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.retrievers.visual import build_visual_retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Query, RetrievalResult

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS = ROOT / "data" / "mmlongbench" / "documents"
GOLDEN = ROOT / "data" / "golden" / "mmlongbench-v1.yaml"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
LLM = "gemma3:4b"

# Pre-ingested 20-doc MMLongBench corpus (2361 chunks, full payload incl. text,
# from study_routing.py). Reused instead of re-ingesting: in-process docling
# ingest of 20 docs (incl. 55- and 166-page PDFs) exhausts native memory and
# segfaults. scroll_chunks() rebuilds BM25 + chunks_by_id in process (the
# minimal-payload mmlb_*_20 collections store only chunk_id+paper_id, no text,
# so they can't be scrolled back).
COLLECTION = "routing_study"

# Depth to retrieve / persist per leg. 50 = candidate-pool / rerank-input
# ceiling on the text leg; the visual leg has no pool cap (it MaxSim-scores
# every page) so its coverage@50 is clean.
DEPTH = 50

# Exact 20 docs from data/eval/baseline-mmlongbench-router.json config.paper_ids
# (identical to study_routing.py — keep in sync).
DOC_IDS = [
    "05-03-18-political-release", "0b85477387a9d0cc33fca0f4becaa0e5",
    "0e94b4197b10096b1f4c699701570fbf", "11-21-16-Updated-Post-Election-Release",
    "12-15-15-ISIS-and-terrorism-release-final", "2005.12872v3", "2021-Apple-Catalog",
    "2023.acl-long.386", "2023.findings-emnlp.248", "2024.ug.eprospectus",
    "2210.02442v1", "2303.05039v2", "2303.08559v2", "2305.13186v3", "2305.14160v4",
    "2306.05425v1", "2307.09288v2", "2309.17421v2", "2310.05634v2", "2310.07609v1",
]

OUT = ROOT / "data" / "eval" / "runs" / f"depth50-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


async def _evict_ollama() -> None:
    """Free VRAM before ColQwen2 (8 GB card). Mirrors study_routing / eval_run."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            ps = await c.get(f"{OLLAMA}/api/ps")
            for m in ps.json().get("models") or []:
                name = m["name"]
                ep = "embeddings" if ("bge-m3" in name or "embed" in name) else "generate"
                body = {"model": name, "keep_alive": 0}
                if ep == "generate":
                    body |= {"prompt": "x", "stream": False, "options": {"num_predict": 1}}
                else:
                    body |= {"prompt": "x"}
                await c.post(f"{OLLAMA}/api/{ep}", json=body)
        log("evicted resident Ollama models")
    except Exception as e:
        log(f"eviction skipped ({e!r})")


def _recall10_macro(rescored: dict[str, Any], subset: set[str] | None) -> tuple[int, float]:
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
    log(f"Depth-50 diagnostic start. out={OUT}  DEPTH={DEPTH}")

    pdfs = [(d, DOCS / f"{d}.pdf") for d in DOC_IDS]
    missing = [d for d, p in pdfs if not p.exists()]
    if missing:
        log(f"ABORT: missing MMLongBench docs: {missing}")
        return

    # --- load the pre-ingested 20-doc corpus from its persistent Qdrant
    # collection. Re-ingesting in-process exhausts docling/native memory on the
    # big PDFs (segfault, see COLLECTION note); the chunks already exist, so
    # scroll them back and rebuild BM25 + chunks_by_id in process. ---
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

    # candidate_pool bumped to DEPTH so the text leg's dense+BM25 pool isn't
    # narrower than the depth we report (default 50 would equal DEPTH; set
    # explicit so a future DEPTH>50 stays honest). rerank_input_size likewise.
    text = PipelineRetriever(
        embedder=embedder, vectorstore=vs, bm25=bm25,
        chunks_by_id=chunks_by_id, reranker=BgeReranker(length_norm=True),
        candidate_pool=DEPTH, rerank_input_size=DEPTH,
    )

    gs = load_golden_set(GOLDEN)
    golden_doc = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    log(f"golden: {len(gs.queries)} queries")

    # --- PASS 1: text leg for ALL queries at DEPTH, BEFORE ColQwen2 loads.
    # Co-residency of the 568M bge-reranker and ColQwen2 on 8GB makes the
    # reranker ~25x slower (measured: 56s vs 2s/query). Running the whole text
    # pass first, while ColQwen2 is not yet resident, keeps it fast. This pass
    # doubles as the sanity gate (score its top-10). ---
    text_results: dict[str, list[RetrievalResult]] = {}
    text_pq = []
    for q in gs.queries:
        res = await text.retrieve(Query(text=q.text, top_k=DEPTH))
        text_results[q.query_id] = res
        text_pq.append({
            "query_id": q.query_id, "category": q.category,
            "retrieved_chunk_ids": [r.chunk_id for r in res[:10]],
        })
    n, r10 = _recall10_macro(rescore({"per_query": text_pq}, golden_doc), None)
    log(f"SANITY text-only@10: n={n} recall@10={r10:.4f} (committed text ref 0.5545)")
    if r10 < 0.45:
        log("ABORT: text-only recall@10 < 0.45 — corpus/scoring wrong.")
        return
    log(f"PASS 1 (text) done, {len(text_results)} queries. Loading ColQwen2.")

    # --- visual leg (ColQwen2) — GPU-heavy. Built AFTER the text pass so the
    # two large models are never resident together. ---
    await _evict_ollama()
    pages_by_paper = {}
    for did, pdf in pdfs:
        rendered = render_pages(did, pdf, out_dir=ROOT / "data" / "pages", dpi=150)
        pages_by_paper[did] = [(p.page_number, p.image_path) for p in rendered]
    log(f"rendered pages for {len(pages_by_paper)} papers")
    visual = await build_visual_retriever(
        pages_by_paper, model_name="vidore/colqwen2-v1.0", device="cuda"
    )
    log("ColQwen2 visual leg built")

    clf = LLMQueryClassifier(
        llm=OllamaChatClient(base_url=OLLAMA), model=LLM,
        prompt=load_prompt_by_name("classify_query"),
    )
    # RoutingRetriever only to reuse its real RRF fusion (_fuse_page_level).
    router = RoutingRetriever(text=text, visual=visual, classifier=clf)

    # --- PASS 2: visual leg for ALL queries; fuse with the stored text results.
    # No text retrieval here, so the reranker never runs while ColQwen2 is
    # resident — the contention that made the interleaved version 25x slower. ---
    per_query = []
    for i, q in enumerate(gs.queries):
        visual_res = await visual.retrieve(Query(text=q.text, top_k=DEPTH))
        text_res = text_results[q.query_id]
        fused_res = router._fuse_page_level(text_res, visual_res, top_k=DEPTH)
        per_query.append({
            "query_id": q.query_id,
            "category": q.category,
            "text_top50": [r.chunk_id for r in text_res],
            "visual_top50": [r.chunk_id for r in visual_res],
            "fused_top50": [r.chunk_id for r in fused_res],
        })
        if (i + 1) % 30 == 0:
            log(f"  visual+fuse: {i + 1}/{len(gs.queries)}")
    log("depth-50 run complete")

    (OUT / "depth50.json").write_text(
        json.dumps({"depth": DEPTH, "per_query": per_query}, indent=2), encoding="utf-8"
    )
    log(f"wrote {OUT / 'depth50.json'}")

    # --- self-check: fused TOP-10 recall must reproduce the committed router ---
    # (this is the dumbest sanity check: depth-50 must observe the same system)
    fused10_pq = [
        {"query_id": r["query_id"], "category": r["category"],
         "retrieved_chunk_ids": r["fused_top50"][:10]}
        for r in per_query
    ]
    rescored10 = rescore({"per_query": fused10_pq}, golden_doc)
    n_all, r10_all = _recall10_macro(rescored10, None)
    n_fig, r10_fig = _recall10_macro(rescored10, {"figure"})
    log(f"SELF-CHECK fused@10: all n={n_all} recall@10={r10_all:.4f} "
        f"(committed router = 0.7461)")
    log(f"SELF-CHECK fused@10: figure n={n_fig} recall@10={r10_fig:.4f} "
        f"(committed router figure = 0.7578)")
    if abs(r10_all - 0.7461) > 0.02:
        log("WARNING: fused@10 recall drifted >0.02 from committed router — "
            "the depth-50 run is NOT observing the same system. Investigate "
            "before trusting the decomposition.")
    else:
        log("SELF-CHECK OK — depth-50 fused@10 reproduces the shipped router.")

    print(
        "\nNext: decompose the figure subset with\n"
        f"  .venv/Scripts/python.exe -m scripts.experiments.diagnose_figure_misses \\\n"
        f"      --run {OUT / 'depth50.json'} --golden {GOLDEN} "
        "--k-recall 10 --k-coverage 50\n",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
