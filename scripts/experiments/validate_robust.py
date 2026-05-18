"""Empirically prove robust-v1 is routing-fair (ADR 0015).

Fair ⇔ a lazy policy cannot win: always-text must fail the visual buckets,
always-visual must fail the text bucket, and bucket-oracle (route by the
TRUE evidence label) must beat BOTH by a clear margin. No LLM classifier —
the oracle uses the golden's own bucket label, so this run is keyless.

Spans two corpora, so retrieval is **paper-filtered** (each query carries
paper_id) and results are post-filtered to the query's own document before
page-scoring — raw page numbers collide across docs otherwise (ADR 0015).

Sanity gate: after ingest, always-text on the text bucket must score
clearly non-zero, else ABORT (ingest/filter broken) — don't burn the night.

Output: data/eval/runs/robust-validate-<ts>/REPORT.md + verdict.
"""

from __future__ import annotations

import contextlib
import re
import sys
import time
from pathlib import Path

import yaml

from scripts.rescore_mmlb_pages import rescore
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.ingestion.pipeline import ingest_paper
from src.ingestion.visual import render_pages
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.visual import build_visual_retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper, Query

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden" / "robust-v1.yaml"
ARXIV = ROOT / "data" / "papers"
MMLB = ROOT / "data" / "mmlongbench" / "documents"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
VISUAL_BUCKETS = {"figure", "table", "mixed"}
_BUCKET = re.compile(r"bucket=(\w+)")
OUT = ROOT / "data" / "eval" / "runs" / f"robust-validate-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _bucket(note: str | None) -> str:
    m = _BUCKET.search(note or "")
    return m.group(1) if m else "text"


def _pdf_for(paper_id: str) -> Path | None:
    a = ARXIV / f"{paper_id}.pdf"
    m = MMLB / f"{paper_id}.pdf"
    return a if a.exists() else (m if m.exists() else None)


def macro(rescored: dict, bucket: str | None, buckets: dict[str, str]) -> tuple[float, int]:
    vals = [
        float(pq["retrieval"]["recall_at_10"])
        for pq in rescored["per_query"]
        if (pq.get("retrieval") or {}).get("recall_at_10") is not None
        and (bucket is None or buckets.get(pq["query_id"]) == bucket)
    ]
    return (sum(vals) / len(vals) if vals else 0.0, len(vals))


async def _score(policy, queries, text, visual, golden_doc):
    per_query = []
    for i, q in enumerate(queries):
        pid = q.paper_id
        rq = Query(text=q.text, top_k=10, filters={"paper_id": pid})
        if policy == "always-text":
            res = await text.retrieve(rq)
        elif policy == "always-visual":
            res = await visual.retrieve(Query(text=q.text, top_k=50))
        else:  # oracle: route by the true bucket label
            b = _bucket(q.note)
            if b in VISUAL_BUCKETS:
                res = await visual.retrieve(Query(text=q.text, top_k=50))
            else:
                res = await text.retrieve(rq)
        # post-filter to the query's own document, then top-10 (ADR 0015)
        ids = [r.chunk_id for r in res if r.chunk_id.startswith(f"{pid}::")][:10]
        per_query.append(
            {"query_id": q.query_id, "category": q.category, "retrieved_chunk_ids": ids}
        )
        if (i + 1) % 30 == 0:
            log(f"  {policy}: {i + 1}/{len(queries)}")
    return rescore({"per_query": per_query}, golden_doc)


async def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    log("robust-v1 fairness validation start")

    gs = load_golden_set(GOLDEN)
    golden_doc = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    buckets = {q.query_id: _bucket(q.note) for q in gs.queries}
    paper_ids = sorted({q.paper_id for q in gs.queries})
    pdfs = {pid: _pdf_for(pid) for pid in paper_ids}
    missing = [pid for pid, p in pdfs.items() if p is None]
    if missing:
        log(f"ABORT: missing PDFs for {missing}")
        return
    log(f"{len(gs.queries)} queries over {len(paper_ids)} docs; ingesting")

    embedder = OllamaBgeEmbedder(base_url=OLLAMA)
    vs = QdrantVectorStore(url=QDRANT, collection_name="robust_validate", dim=embedder.dim)
    await vs.ensure_collection()
    bm25 = Bm25Index()
    chunks_by_id = {}
    for pid, pdf in pdfs.items():
        ing = await ingest_paper(
            paper=Paper(paper_id=pid, title=pid, pdf_path=pdf),
            embedder=embedder, vectorstore=vs, bm25=bm25,
            contextualizer_llm=None, contextualizer_model=None,
            contextualizer_concurrency=1,
            extract_figures_enabled=False, extract_tables_enabled=False,
            vlm_captioner=None,
        )
        for c in ing.chunks:
            chunks_by_id[c.chunk_id] = c
        log(f"  ingested {pid}: {ing.chunk_count}")
    text = PipelineRetriever(
        embedder=embedder, vectorstore=vs, bm25=bm25,
        chunks_by_id=chunks_by_id, reranker=BgeReranker(length_norm=True),
    )

    # SANITY: always-text on the text bucket must be clearly non-zero.
    tbucket = [q for q in gs.queries if buckets[q.query_id] == "text"]
    s = await _score("always-text", tbucket, text, None, golden_doc)
    sane, n = macro(s, None, buckets)
    log(f"SANITY always-text on text bucket (n={n}): recall@10={sane:.4f}")
    if sane < 0.30:
        log("ABORT: text-leg ~0 on text bucket — ingest/paper-filter broken.")
        return
    log("SANITY OK — proceeding to ColQwen2 + full run.")

    # visual leg over BOTH corpora's pages
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as c:
            ps = await c.get(f"{OLLAMA}/api/ps")
            for mm in ps.json().get("models") or []:
                nm = mm["name"]
                ep = "embeddings" if ("bge-m3" in nm or "embed" in nm) else "generate"
                body = {"model": nm, "keep_alive": 0}
                body |= {"prompt": "x"} if ep == "embeddings" else {
                    "prompt": "x", "stream": False, "options": {"num_predict": 1}}
                await c.post(f"{OLLAMA}/api/{ep}", json=body)
        log("evicted Ollama models")
    except Exception as e:
        log(f"eviction skipped ({e!r})")
    pages_by_paper = {}
    for pid, pdf in pdfs.items():
        rendered = render_pages(pid, pdf, out_dir=ROOT / "data" / "pages", dpi=150)
        pages_by_paper[pid] = [(p.page_number, p.image_path) for p in rendered]
    visual = await build_visual_retriever(
        pages_by_paper, model_name="vidore/colqwen2-v1.0", device="cuda"
    )
    log("ColQwen2 built")

    results = {}
    for policy in ("always-text", "always-visual", "oracle"):
        results[policy] = await _score(policy, gs.queries, text, visual, golden_doc)
        log(f"done {policy}")

    bset = ["text", "figure", "table", "mixed"]
    lines = ["# robust-v1 fairness validation — recall@10 (page-level, paper-filtered)\n",
             "| policy | overall | " + " | ".join(bset) + " |",
             "|---|---|" + "---|" * len(bset)]
    ov = {}
    for p in ("always-text", "always-visual", "oracle"):
        o, _ = macro(results[p], None, buckets)
        ov[p] = o
        cells = " | ".join(f"{macro(results[p], b, buckets)[0]:.3f}" for b in bset)
        lines.append(f"| {p} | **{o:.4f}** | {cells} |")
    margin = ov["oracle"] - max(ov["always-text"], ov["always-visual"])
    verdict = (
        "PASS — routing-fair: oracle beats both lazy policies by "
        f"{margin:+.4f} recall@10"
        if margin >= 0.05
        else f"FAIL — set NOT routing-fair (oracle margin only {margin:+.4f}); "
        "a lazy policy ~matches oracle, rebalance needed"
    )
    lines += ["", verdict,
              "Expected if fair: always-text strong on `text` weak on visual; "
              "always-visual the reverse; oracle strong on all."]
    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    log("REPORT.md written")
    print("\n===REPORT===\n" + report, flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
