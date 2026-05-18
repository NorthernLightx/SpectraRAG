"""Extensive routing study on the CORRECT MMLongBench corpus.

Delivers: the best routing config for top retrieval accuracy at good speed.

Compares four policies on the 20 MMLongBench documents the committed
`baseline-mmlongbench.json` uses (NOT the v3 arXiv papers -- that was the
earlier corpus mistake):

  1. text-only       PipelineRetriever (dense+BM25+RRF+rerank)
  2. regex-router    RoutingRetriever, regex classify_query (the shipped default)
  3. llm-router      RoutingRetriever, LLMQueryClassifier (Ollama gemma3:4b)
  4. oracle-router   route by the golden's true category (the ceiling)

Scoring is page-level via the repo's own `rescore` (MMLongBench labels are
page-level; chunk-level scoring is 0.0 by construction -- the trap from
before). Hard sanity gate: text-only page-scored recall@10 must land in a
believable band or the run ABORTS (catches corpus/scoring errors before
burning the night). Retrieval-only: page recall@10/nDCG@5 is
generator-independent and is the decisive routing signal; generation/judge
would 4x the cost for no extra routing signal.

Output: data/eval/runs/routing-study-<ts>/  -- per-policy JSONs, REPORT.md
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import yaml

from src.eval.golden_set import load_golden_set
from src.llm.ollama_chat import OllamaChatClient
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.classifier_llm import LLMQueryClassifier
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.retrievers.visual import build_visual_retriever
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.ingestion.pipeline import ingest_paper
from src.ingestion.visual import render_pages
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper, Query
from scripts.rescore_mmlb_pages import rescore

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "mmlongbench" / "documents"
GOLDEN = ROOT / "data" / "golden" / "mmlongbench-v1.yaml"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
LLM = "gemma3:4b"
HYBRID = {"figure", "table", "multi_hop"}
# Exact 20 docs from data/eval/baseline-mmlongbench.json config.paper_ids
DOC_IDS = [
    "05-03-18-political-release", "0b85477387a9d0cc33fca0f4becaa0e5",
    "0e94b4197b10096b1f4c699701570fbf", "11-21-16-Updated-Post-Election-Release",
    "12-15-15-ISIS-and-terrorism-release-final", "2005.12872v3", "2021-Apple-Catalog",
    "2023.acl-long.386", "2023.findings-emnlp.248", "2024.ug.eprospectus",
    "2210.02442v1", "2303.05039v2", "2303.08559v2", "2305.13186v3", "2305.14160v4",
    "2306.05425v1", "2307.09288v2", "2309.17421v2", "2310.05634v2", "2310.07609v1",
]

OUT = ROOT / "data" / "eval" / "runs" / f"routing-study-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


async def _evict_ollama() -> None:
    """Free VRAM before ColQwen2 (8 GB card). Mirrors eval_run's dance."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            ps = await c.get(f"{OLLAMA}/api/ps")
            for m in (ps.json().get("models") or []):
                name = m["name"]
                ep = "embeddings" if ("bge-m3" in name or "embed" in name) else "generate"
                body = {"model": name, "keep_alive": 0}
                if ep == "generate":
                    body |= {"prompt": "x", "stream": False, "options": {"num_predict": 1}}
                else:
                    body |= {"prompt": "x"}
                await c.post(f"{OLLAMA}/api/{ep}", json=body)
        log("evicted resident Ollama models")
    except Exception as e:  # noqa: BLE001
        log(f"eviction skipped ({e!r})")


def macro(rescored: dict, subset: set[str] | None) -> dict[str, float]:
    rows = [
        pq for pq in rescored["per_query"]
        if pq.get("category") != "out_of_corpus"
        and (pq.get("retrieval") or {}).get("recall_at_10") is not None
        and (subset is None or pq.get("category") in subset)
    ]
    if not rows:
        return {"n": 0, "ndcg_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    n = len(rows)
    return {
        "n": n,
        "ndcg_at_5": sum(float(r["retrieval"]["ndcg_at_5"]) for r in rows) / n,
        "recall_at_10": sum(float(r["retrieval"]["recall_at_10"]) for r in rows) / n,
        "mrr": sum(float(r["retrieval"]["mrr"]) for r in rows) / n,
    }


async def score(policy: str, text, visual, clf, queries, golden_doc) -> tuple[dict, float]:
    """Run every query under `policy`, page-score, return (rescored, clf_seconds)."""
    router = None
    if policy == "regex-router":
        router = RoutingRetriever(text=text, visual=visual)
    elif policy == "llm-router":
        router = RoutingRetriever(text=text, visual=visual, classifier=clf)
    per_query, clf_secs = [], 0.0
    for i, q in enumerate(queries):
        rq = Query(text=q.text, top_k=10)
        if policy == "text-only":
            res = await text.retrieve(rq)
        elif policy == "oracle-router":
            leg = visual if q.category in HYBRID else text
            res = await leg.retrieve(rq)
        else:
            t0 = time.perf_counter()
            res = await router.retrieve(rq)
            clf_secs += time.perf_counter() - t0
        per_query.append({
            "query_id": q.query_id, "category": q.category,
            "retrieved_chunk_ids": [r.chunk_id for r in res],
        })
        if (i + 1) % 30 == 0:
            log(f"  {policy}: {i + 1}/{len(queries)}")
    rescored = rescore({"per_query": per_query}, golden_doc)
    (OUT / f"{policy}.json").write_text(json.dumps(rescored, indent=2), encoding="utf-8")
    return rescored, clf_secs


async def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"Routing study start. out={OUT}")

    pdfs = [(d, DOCS / f"{d}.pdf") for d in DOC_IDS]
    missing = [d for d, p in pdfs if not p.exists()]
    if missing:
        log(f"ABORT: missing MMLongBench docs: {missing}")
        return

    # --- ingest the correct 20-doc corpus once ---
    embedder = OllamaBgeEmbedder(base_url=OLLAMA)
    vs = QdrantVectorStore(url=QDRANT, collection_name="routing_study", dim=embedder.dim)
    await vs.ensure_collection()
    bm25 = Bm25Index()
    chunks_by_id = {}
    for did, pdf in pdfs:
        ing = await ingest_paper(
            paper=Paper(paper_id=did, title=did, pdf_path=pdf),
            embedder=embedder, vectorstore=vs, bm25=bm25,
            contextualizer_llm=None, contextualizer_model=None,
            contextualizer_concurrency=1,
            extract_figures_enabled=False, extract_tables_enabled=False,
            vlm_captioner=None,
        )
        for c in ing.chunks:
            chunks_by_id[c.chunk_id] = c
        log(f"ingested {did}: {ing.chunk_count} chunks")
    text = PipelineRetriever(
        embedder=embedder, vectorstore=vs, bm25=bm25,
        chunks_by_id=chunks_by_id, reranker=BgeReranker(length_norm=True),
    )

    gs = load_golden_set(GOLDEN)
    golden_doc = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    log(f"golden: {len(gs.queries)} queries")

    # --- SANITY GATE: text-only must reproduce a believable recall@10 ---
    t_rescored, _ = await score("text-only", text, None, None, gs.queries, golden_doc)
    t_all = macro(t_rescored, None)
    log(f"SANITY text-only: n={t_all['n']} recall@10={t_all['recall_at_10']:.4f} "
        f"ndcg@5={t_all['ndcg_at_5']:.4f} (results.md ref: ~0.685 / ~0.590)")
    if t_all["recall_at_10"] < 0.45:
        log("ABORT: text-only recall@10 < 0.45 — corpus/scoring still wrong, "
            "not burning the night (this is the gate that catches the earlier bug).")
        return
    log("SANITY OK — corpus + page-scoring correct. Proceeding.")

    # --- visual leg (ColQwen2) — the GPU-heavy part ---
    await _evict_ollama()
    pages_by_paper = {}
    for did, pdf in pdfs:
        rendered = render_pages(did, pdf, out_dir=ROOT / "data" / "pages", dpi=150)
        pages_by_paper[did] = [(p.page_number, p.image_path) for p in rendered]
        log(f"rendered {did}: {len(rendered)} pages")
    visual = await build_visual_retriever(
        pages_by_paper, model_name="vidore/colqwen2-v1.0", device="cuda"
    )
    log("ColQwen2 visual leg built")

    clf = LLMQueryClassifier(
        llm=OllamaChatClient(base_url=OLLAMA), model=LLM,
        prompt=load_prompt_by_name("classify_query"),
    )

    results = {"text-only": (t_rescored, 0.0)}
    for policy in ("regex-router", "llm-router", "oracle-router"):
        try:
            results[policy] = await score(policy, text, visual, clf, gs.queries, golden_doc)
            log(f"done {policy}")
        except Exception as e:  # noqa: BLE001
            log(f"FAILED {policy}: {type(e).__name__}: {e}")

    # --- report (plain) ---
    lines = ["# MMLongBench routing study — page-level retrieval\n",
             "Correct 20-doc corpus. Higher = better. Decisive metric: recall@10.\n",
             "| policy | n | recall@10 | nDCG@5 | MRR | figure recall@10 | table recall@10 | clf s/query |",
             "|---|---|---|---|---|---|---|---|"]
    for policy, (rs, secs) in results.items():
        a = macro(rs, None)
        fig = macro(rs, {"figure"})
        tab = macro(rs, {"table"})
        per = f"{secs / max(1, len(gs.queries)):.2f}" if secs else "-"
        lines.append(
            f"| {policy} | {a['n']} | {a['recall_at_10']:.4f} | {a['ndcg_at_5']:.4f} "
            f"| {a['mrr']:.4f} | {fig['recall_at_10']:.4f} | {tab['recall_at_10']:.4f} | {per} |"
        )
    base = macro(results["text-only"][0], None)["recall_at_10"]
    best = max(results, key=lambda k: macro(results[k][0], None)["recall_at_10"])
    bestv = macro(results[best][0], None)["recall_at_10"]
    lines += [
        "",
        f"Text-only recall@10 = {base:.4f}. Best = **{best}** at {bestv:.4f} "
        f"(+{(bestv - base) / base * 100:.1f}% vs text-only).",
        "Oracle = ceiling; gap (oracle - llm-router) = headroom a perfect classifier still leaves.",
        "Speed: 'clf s/query' is the only added cost vs text-only; cascade (ADR 0010) "
        "can skip the visual call on confident-text queries to protect it further.",
    ]
    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    log("REPORT.md written")
    print("\n===REPORT===\n" + report, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
