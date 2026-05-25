"""Cross-corpus validation of the ADR 0023 visual-fusion weight on golden v3.

ADR 0023 shipped a visual-leg RRF weight (`visual_fusion_weight`, default 1.0 =
OFF) that lifts MMLongBench figure recall@10 0.73 -> 0.81. MMLongBench is ~93 %
visual; ADR 0013 warns a TEXT-HEAVY corpus may not want a visual bias. This
driver runs the same depth-50 two-pass measurement on the text-heavy arXiv
corpus (golden v3) to decide the safe default.

Method (mirrors `scripts/experiments/diagnose_depth50_run.py`, two passes,
never co-resident on the 8 GB card):

  PASS 1 (text, ColQwen2 NOT yet loaded): load the text leg from the
    `eval_phase32_router` Qdrant collection via scroll_chunks + BM25 rebuild +
    PipelineRetriever(reranker=BgeReranker(length_norm=True), pool=50,
    rerank_input=50). Run text retrieval top_k=50 for every v3 query.
  PASS 2 (visual): evict Ollama, render the 12 v3 papers, build the ColQwen2
    visual leg, run it top_k=50 for every query.
  RE-FUSE (CPU): feed each query's text_top50 / visual_top50 through the REAL
    RoutingRetriever._fuse_page_level at w in {1, 2, 3, 5} and score the
    figure+table subsets.

Two divergences from the MMLongBench diagnose script, both deliberate:

  * Paper-id filter ON both legs. The shipped v3 baseline (data/eval/
    baseline.json config.paper_id_filter == True) scopes retrieval to
    GoldenQuery.paper_id; the `eval_phase32_router` collection holds 20 papers
    (12 v3 + 8 distractors), so an unfiltered top-50 text leg would be polluted
    by wrong-paper hits the real router never sees. Filtering reproduces the
    router this experiment is supposed to inform. The MMLongBench script is a
    different (unfiltered) config.
  * Metric is dual. v3 golden is CHUNK-scored (`relevant_chunk_ids`), unlike
    MMLongBench (page-scored `relevant_pages`). But the knob lives in
    `_fuse_page_level`, which fuses and returns at PAGE granularity. So the
    metric that isolates the fusion lever is PAGE-level recall (collapse gold
    chunk-ids to (paper,page)); that is the primary report. Chunk-level recall
    (v3's native granularity, via src.eval.metrics_retrieval) is reported as a
    cross-check, and is structurally depressed for three reasons noted at
    runtime: (a) page fusion collapses same-page gold chunks (q25/q26/q31 have
    two gold chunks on one page -> chunk recall caps at 0.5); (b) the fused
    representative is the best text chunk on a page, which need not be the exact
    gold chunk; (c) visual-only pages carry a `::page` id that can never
    chunk-match a `::cM` gold id.

Run on the RTX 3070 box (Ollama + Qdrant up, ColQwen2 weights cached):
    .venv/Scripts/python.exe -m scripts.experiments.validate_v3_visual_fusion_weight
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.ingestion.visual import render_pages
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Query, RetrievalResult

ROOT = Path(__file__).resolve().parent.parent.parent
PAPERS = ROOT / "data" / "papers"
GOLDEN = ROOT / "data" / "golden" / "v3.yaml"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
COLLECTION = "eval_phase32_router"
DEPTH = 50
WEIGHTS = (1.0, 2.0, 3.0, 5.0)
SCORED_CATEGORIES = ("figure", "table")

OUT = ROOT / "data" / "eval" / "runs" / f"v3-fusionweight-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"

_PAGE_RE = re.compile(r"::p(\d+)")
Page = tuple[str, int]


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _page_of(chunk_id: str) -> Page | None:
    match = _PAGE_RE.search(chunk_id)
    if match is None:
        return None
    return chunk_id.split("::", 1)[0], int(match.group(1))


def _pages_in_rank(chunk_ids: list[str]) -> list[str]:
    """Rank-ordered page-id list ('paper::pN'), each page once at first appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        page = _page_of(cid)
        if page is None:
            continue
        page_id = f"{page[0]}::p{page[1]}"
        if page_id not in seen:
            seen.add(page_id)
            out.append(page_id)
    return out


async def _evict_ollama() -> None:
    """Free VRAM before ColQwen2 (8 GB card). Mirrors diagnose_depth50_run."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA}/api/ps")
            for model in resp.json().get("models") or []:
                name = model["name"]
                endpoint = "embeddings" if ("bge-m3" in name or "embed" in name) else "generate"
                body: dict[str, Any] = {"model": name, "keep_alive": 0}
                if endpoint == "generate":
                    body |= {"prompt": "x", "stream": False, "options": {"num_predict": 1}}
                else:
                    body |= {"prompt": "x"}
                await client.post(f"{OLLAMA}/api/{endpoint}", json=body)
        log("evicted resident Ollama models")
    except Exception as exc:
        log(f"eviction skipped ({exc!r})")


def _filtered_query(text: str, paper_id: str) -> Query:
    """Query scoped to one paper — mirrors the v3 baseline's paper_id_filter=True."""
    return Query(text=text, top_k=DEPTH, filters={"paper_id": paper_id})


def _score_page_level(refused_ids: list[str], gold_pages: set[str]) -> tuple[float, float, float]:
    pages = _pages_in_rank(refused_ids)
    return (
        recall_at_k(gold_pages, pages, k=10),
        ndcg_at_k(gold_pages, pages, k=5),
        reciprocal_rank(gold_pages, pages),
    )


def _score_chunk_level(refused_ids: list[str], gold_chunks: set[str]) -> tuple[float, float, float]:
    return (
        recall_at_k(gold_chunks, refused_ids, k=10),
        ndcg_at_k(gold_chunks, refused_ids, k=5),
        reciprocal_rank(gold_chunks, refused_ids),
    )


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"v3 visual-fusion-weight validation start. out={OUT} DEPTH={DEPTH}")

    golden = load_golden_set(GOLDEN)
    fig_table = [q for q in golden.queries if q.category in SCORED_CATEGORIES]
    paper_ids = sorted({q.paper_id for q in golden.queries if q.paper_id})
    log(
        f"golden v3: {len(golden.queries)} queries; {len(fig_table)} figure/table; "
        f"{len(paper_ids)} papers"
    )

    missing_pdfs = [p for p in paper_ids if not (PAPERS / f"{p}.pdf").exists()]
    if missing_pdfs:
        log(f"ABORT: missing PDFs for {missing_pdfs}")
        return

    # --- text leg from the persistent collection (re-ingest would OOM on the
    # long PDFs; scroll + BM25 rebuild in process). ---
    embedder = OllamaBgeEmbedder(base_url=OLLAMA)
    vectorstore = QdrantVectorStore(url=QDRANT, collection_name=COLLECTION, dim=embedder.dim)
    chunks = await vectorstore.scroll_chunks()
    if not chunks:
        log(f"ABORT: collection {COLLECTION!r} empty or missing.")
        return
    bm25 = Bm25Index()
    bm25.add(chunks)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    log(
        f"loaded {len(chunks)} chunks from {COLLECTION!r} "
        f"({len({c.paper_id for c in chunks})} distinct papers)"
    )

    text = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        reranker=BgeReranker(length_norm=True),
        candidate_pool=DEPTH,
        rerank_input_size=DEPTH,
    )

    # --- PASS 1: text leg for every fig/table query, BEFORE ColQwen2 loads. ---
    text_results: dict[str, list[RetrievalResult]] = {}
    text_page_recall: list[float] = []
    for query in fig_table:
        assert query.paper_id is not None
        res = await text.retrieve(_filtered_query(query.text, query.paper_id))
        text_results[query.query_id] = res
        gold_pages = {f"{p[0]}::p{p[1]}" for c in query.relevant_chunk_ids if (p := _page_of(c))}
        text_page_recall.append(
            recall_at_k(gold_pages, _pages_in_rank([r.chunk_id for r in res]), k=10)
        )
    text_recall = _avg(text_page_recall)
    log(
        f"SANITY text-only page recall@10 (figure+table, n={len(text_page_recall)}): "
        f"{text_recall:.4f} — v3 text is strong, expect high."
    )
    if text_recall < 0.5:
        log(
            "WARNING: text-only page recall@10 < 0.5 — paper filter or scoring suspect. "
            "Continuing, but inspect before trusting fusion deltas."
        )
    log("PASS 1 (text) done. Evicting Ollama, loading ColQwen2.")

    # --- PASS 2: visual leg (ColQwen2). Built AFTER the text pass so the two
    # large models are never resident together. ---
    await _evict_ollama()
    pages_by_paper: dict[str, list[tuple[int, Path]]] = {}
    for paper_id in paper_ids:
        rendered = render_pages(
            paper_id, PAPERS / f"{paper_id}.pdf", out_dir=ROOT / "data" / "pages", dpi=150
        )
        pages_by_paper[paper_id] = [(p.page_number, p.image_path) for p in rendered]
    log(
        f"rendered pages for {len(pages_by_paper)} papers "
        f"({sum(len(v) for v in pages_by_paper.values())} pages total)"
    )

    from src.rag.retrievers.visual import build_visual_retriever

    visual = await build_visual_retriever(
        pages_by_paper, model_name="vidore/colqwen2-v1.0", device="cuda"
    )
    log("ColQwen2 visual leg built")

    visual_results: dict[str, list[RetrievalResult]] = {}
    visual_page_recall: list[float] = []
    for query in fig_table:
        assert query.paper_id is not None
        res = await visual.retrieve(_filtered_query(query.text, query.paper_id))
        visual_results[query.query_id] = res
        gold_pages = {f"{p[0]}::p{p[1]}" for c in query.relevant_chunk_ids if (p := _page_of(c))}
        visual_page_recall.append(
            recall_at_k(gold_pages, _pages_in_rank([r.chunk_id for r in res]), k=10)
        )
    log(
        f"visual-only page recall@10 (figure+table, n={len(visual_page_recall)}): "
        f"{_avg(visual_page_recall):.4f}"
    )

    # --- persist the legs (audit + re-run without GPU) ---
    per_query_dump = [
        {
            "query_id": q.query_id,
            "category": q.category,
            "paper_id": q.paper_id,
            "relevant_chunk_ids": list(q.relevant_chunk_ids),
            "text_top50": [r.chunk_id for r in text_results[q.query_id]],
            "visual_top50": [r.chunk_id for r in visual_results[q.query_id]],
        }
        for q in fig_table
    ]
    (OUT / "legs.json").write_text(
        json.dumps(
            {"depth": DEPTH, "collection": COLLECTION, "per_query": per_query_dump}, indent=2
        ),
        encoding="utf-8",
    )
    log(f"wrote {OUT / 'legs.json'}")

    # --- chunk-level confound report: which gold ids can the text leg never hit? ---
    all_ids = set(chunks_by_id)
    unreachable = [
        (q.query_id, cid) for q in fig_table for cid in q.relevant_chunk_ids if cid not in all_ids
    ]
    same_page = [
        q.query_id
        for q in fig_table
        if len({f"{p[0]}::p{p[1]}" for c in q.relevant_chunk_ids if (p := _page_of(c))}) == 1
        and len(q.relevant_chunk_ids) > 1
    ]
    log(
        f"chunk-level confounds: {len(unreachable)} gold ids absent from the text index "
        f"{[cid for _, cid in unreachable]}; same-page-collision queries (chunk recall "
        f"caps < 1.0): {same_page}"
    )

    # --- RE-FUSE at each weight through the REAL _fuse_page_level. ---
    log("=" * 78)
    log("PAGE-LEVEL metric (primary — matches the page-granularity fusion lever):")
    weight_rows: dict[str, dict[str, Any]] = {}
    for weight in WEIGHTS:
        router = RoutingRetriever(
            text=_NoopRetriever(), visual=_NoopRetriever(), visual_fusion_weight=weight
        )
        for label, subset in (
            ("figure+table", fig_table),
            ("figure", [q for q in fig_table if q.category == "figure"]),
            ("table", [q for q in fig_table if q.category == "table"]),
        ):
            page_recall: list[float] = []
            page_ndcg: list[float] = []
            page_mrr: list[float] = []
            chunk_recall: list[float] = []
            chunk_ndcg: list[float] = []
            chunk_mrr: list[float] = []
            for query in subset:
                # Exercising the real ADR 0023 fusion path directly.
                fused = router._fuse_page_level(
                    text_results[query.query_id],
                    visual_results[query.query_id],
                    top_k=DEPTH,
                )
                refused_ids = [r.chunk_id for r in fused]
                gold_chunks = set(query.relevant_chunk_ids)
                gold_pages = {
                    f"{p[0]}::p{p[1]}" for c in query.relevant_chunk_ids if (p := _page_of(c))
                }
                pr, pn, pm = _score_page_level(refused_ids, gold_pages)
                cr, cn, cm = _score_chunk_level(refused_ids, gold_chunks)
                page_recall.append(pr)
                page_ndcg.append(pn)
                page_mrr.append(pm)
                chunk_recall.append(cr)
                chunk_ndcg.append(cn)
                chunk_mrr.append(cm)
            row = {
                "n": len(subset),
                "page_recall_at_10": _avg(page_recall),
                "page_ndcg_at_5": _avg(page_ndcg),
                "page_mrr": _avg(page_mrr),
                "chunk_recall_at_10": _avg(chunk_recall),
                "chunk_ndcg_at_5": _avg(chunk_ndcg),
                "chunk_mrr": _avg(chunk_mrr),
            }
            weight_rows[f"w={weight}|{label}"] = row
            if label == "figure+table":
                log(
                    f"  w_visual={weight:>4} | {label} (n={row['n']}): "
                    f"recall@10={row['page_recall_at_10']:.4f} "
                    f"nDCG@5={row['page_ndcg_at_5']:.4f} MRR={row['page_mrr']:.4f}"
                )

    log("-" * 78)
    log("per-subset PAGE-level at each weight:")
    for weight in WEIGHTS:
        for label in ("figure", "table"):
            row = weight_rows[f"w={weight}|{label}"]
            log(
                f"  w={weight:>4} {label:<6} n={row['n']:>2}  "
                f"recall@10={row['page_recall_at_10']:.4f} "
                f"nDCG@5={row['page_ndcg_at_5']:.4f} MRR={row['page_mrr']:.4f}"
            )

    log("-" * 78)
    log("CHUNK-LEVEL metric (v3 native; depressed by the confounds logged above):")
    for weight in WEIGHTS:
        row = weight_rows[f"w={weight}|figure+table"]
        log(
            f"  w_visual={weight:>4} | figure+table (n={row['n']}): "
            f"recall@10={row['chunk_recall_at_10']:.4f} "
            f"nDCG@5={row['chunk_ndcg_at_5']:.4f} MRR={row['chunk_mrr']:.4f}"
        )

    (OUT / "weight_sweep.json").write_text(json.dumps(weight_rows, indent=2), encoding="utf-8")
    log(f"wrote {OUT / 'weight_sweep.json'}")
    log("done")


class _NoopRetriever:
    """Satisfies the Retriever protocol; never called (we drive _fuse_page_level)."""

    async def retrieve(self, query: Query) -> list[RetrievalResult]:  # pragma: no cover
        return []


if __name__ == "__main__":
    asyncio.run(main())
