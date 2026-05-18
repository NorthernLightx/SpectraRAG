"""Routing study — llm-router cell, parameterized by classifier model.

Recovers/extends the llm-router policy the main study lost to an Ollama 500
(gemma3:4b /api/chat under VRAM contention with ColQwen2). Fix: decouple
classification from the GPU — classify all 149 queries with NO ColQwen2
loaded (+ retries + safe fallback), cache decisions, only then build
ColQwen2 and route by the cached decisions. Reuses the already-ingested
`routing_study` Qdrant collection and cached page renders.

`--model` selects the Ollama classifier (incl. `:cloud` tags, e.g.
`qwen3-vl:235b-cloud`). Outputs are per-model; the report globs every
`llm-router*.json` so classifier variants sit side by side with the
text-only / regex / oracle baselines from the main study.

Output: <prior-study-dir>/llm-router-<slug>.json + REPORT_routing.md
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path

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
from src.rag.retrievers.visual import build_visual_retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Query

ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "data" / "eval" / "runs" / "routing-study-20260517-161059"
DOCS = ROOT / "data" / "mmlongbench" / "documents"
GOLDEN = ROOT / "data" / "golden" / "mmlongbench-v1.yaml"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
DEFAULT_MODEL = "gemma3:4b"
HYBRID = {"figure", "table", "multi_hop"}
DOC_IDS = [
    "05-03-18-political-release",
    "0b85477387a9d0cc33fca0f4becaa0e5",
    "0e94b4197b10096b1f4c699701570fbf",
    "11-21-16-Updated-Post-Election-Release",
    "12-15-15-ISIS-and-terrorism-release-final",
    "2005.12872v3",
    "2021-Apple-Catalog",
    "2023.acl-long.386",
    "2023.findings-emnlp.248",
    "2024.ug.eprospectus",
    "2210.02442v1",
    "2303.05039v2",
    "2303.08559v2",
    "2305.13186v3",
    "2305.14160v4",
    "2306.05425v1",
    "2307.09288v2",
    "2309.17421v2",
    "2310.05634v2",
    "2310.07609v1",
]
LOG = PRIOR / "recovery.log"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def macro(rescored: dict, subset: set[str] | None) -> dict:
    rows = [
        pq
        for pq in rescored["per_query"]
        if pq.get("category") != "out_of_corpus"
        and (pq.get("retrieval") or {}).get("recall_at_10") is not None
        and (subset is None or pq.get("category") in subset)
    ]
    if not rows:
        return {"n": 0, "recall_at_10": 0.0, "ndcg_at_5": 0.0, "mrr": 0.0}
    n = len(rows)
    return {
        "n": n,
        "recall_at_10": sum(float(r["retrieval"]["recall_at_10"]) for r in rows) / n,
        "ndcg_at_5": sum(float(r["retrieval"]["ndcg_at_5"]) for r in rows) / n,
        "mrr": sum(float(r["retrieval"]["mrr"]) for r in rows) / n,
    }


async def classify_one(clf: LLMQueryClassifier, text: str) -> str:
    for _ in range(3):
        try:
            return await clf.classify(text)
        except Exception as e:
            log(f"  classify retry ({type(e).__name__})")
            await asyncio.sleep(3)
    return "definitional"  # safe fallback -> text leg


def _label(p: Path) -> str:
    s = p.stem  # "llm-router" (the original gemma3:4b run) or "llm-router-<slug>"
    return "llm-router(gemma3:4b)" if s == "llm-router" else f"llm-router({s[11:]})"


async def main(model: str, prompt_name: str) -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    slug = model.replace(":", "_").replace("/", "_")
    if prompt_name != "classify_query":
        slug += f"__{prompt_name}"
    log(f"routing study (classifier={model}, prompt={prompt_name})")

    embedder = OllamaBgeEmbedder(base_url=OLLAMA)
    vs = QdrantVectorStore(url=QDRANT, collection_name="routing_study", dim=embedder.dim)
    chunks = await vs.scroll_chunks()
    if not chunks:
        log("ABORT: routing_study collection empty/missing — would need full re-ingest")
        return
    bm25 = Bm25Index()
    bm25.add(chunks)
    text = PipelineRetriever(
        embedder=embedder,
        vectorstore=vs,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
        reranker=BgeReranker(length_norm=True),
    )
    gs = load_golden_set(GOLDEN)
    golden_doc = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    log(f"text leg rebuilt from {len(chunks)} chunks; {len(gs.queries)} queries")

    # Phase A: classify ALL queries with NO ColQwen2 loaded (the 500 fix).
    clf = LLMQueryClassifier(
        llm=OllamaChatClient(base_url=OLLAMA),
        model=model,
        prompt=load_prompt_by_name(prompt_name),
    )
    decisions: dict[str, str] = {}
    for i, q in enumerate(gs.queries):
        decisions[q.query_id] = await classify_one(clf, q.text)
        if (i + 1) % 30 == 0:
            log(f"  classified {i + 1}/{len(gs.queries)}")
    (PRIOR / f"llm_decisions-{slug}.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    n_hybrid = sum(1 for v in decisions.values() if v in HYBRID)
    log(f"classification done: {n_hybrid}/{len(decisions)} -> visual leg")

    # Phase B: now safe to load ColQwen2.
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as c:
            ps = await c.get(f"{OLLAMA}/api/ps")
            for m in ps.json().get("models") or []:
                nm = m["name"]
                ep = "embeddings" if ("bge-m3" in nm or "embed" in nm) else "generate"
                body = {"model": nm, "keep_alive": 0}
                body |= (
                    {"prompt": "x"}
                    if ep == "embeddings"
                    else {"prompt": "x", "stream": False, "options": {"num_predict": 1}}
                )
                await c.post(f"{OLLAMA}/api/{ep}", json=body)
        log("evicted Ollama models")
    except Exception as e:
        log(f"eviction skipped ({e!r})")

    pages_by_paper = {}
    for did in DOC_IDS:
        rendered = render_pages(did, DOCS / f"{did}.pdf", out_dir=ROOT / "data" / "pages", dpi=150)
        pages_by_paper[did] = [(p.page_number, p.image_path) for p in rendered]
    visual = await build_visual_retriever(
        pages_by_paper, model_name="vidore/colqwen2-v1.0", device="cuda"
    )
    log("ColQwen2 built")

    # Phase C: route by cached decision, score.
    per_query = []
    for i, q in enumerate(gs.queries):
        leg = visual if decisions[q.query_id] in HYBRID else text
        res = await leg.retrieve(Query(text=q.text, top_k=10))
        per_query.append(
            {
                "query_id": q.query_id,
                "category": q.category,
                "retrieved_chunk_ids": [r.chunk_id for r in res],
            }
        )
        if (i + 1) % 30 == 0:
            log(f"  routed {i + 1}/{len(gs.queries)}")
    rescored = rescore({"per_query": per_query}, golden_doc)
    (PRIOR / f"llm-router-{slug}.json").write_text(json.dumps(rescored, indent=2), encoding="utf-8")

    # Report: base policies + every llm-router*.json variant side by side.
    lines = [
        "# MMLongBench routing study — classifier comparison\n",
        "Correct 20-doc corpus, page-level. Higher = better.\n",
        "| policy | n | recall@10 | nDCG@5 | MRR | figure recall@10 |",
        "|---|---|---|---|---|---|",
    ]
    table: dict[str, dict] = {}
    base_files = [("text-only", "text-only.json"), ("regex-router", "regex-router.json")]
    llm_files = sorted(PRIOR.glob("llm-router*.json"))
    rows = (
        [(n, PRIOR / f) for n, f in base_files]
        + [(_label(p), p) for p in llm_files]
        + [("oracle-router", PRIOR / "oracle-router.json")]
    )
    for name, path in rows:
        rs = json.loads(Path(path).read_text(encoding="utf-8"))
        a, fig = macro(rs, None), macro(rs, {"figure"})
        table[name] = a
        lines.append(
            f"| {name} | {a['n']} | {a['recall_at_10']:.4f} | {a['ndcg_at_5']:.4f} "
            f"| {a['mrr']:.4f} | {fig['recall_at_10']:.4f} |"
        )
    base = table["text-only"]["recall_at_10"]
    rgx = table["regex-router"]["recall_at_10"]
    orc = table["oracle-router"]["recall_at_10"]
    lines.append("")
    for name in table:
        if not name.startswith("llm-router"):
            continue
        lr = table[name]["recall_at_10"]
        cap = (lr - rgx) / (orc - rgx) * 100 if orc > rgx else 0.0
        lines.append(
            f"{name}: recall@10 {lr:.4f} (+{(lr - base) / base * 100:.1f}% vs text-only, "
            f"+{(lr - rgx) / rgx * 100:.1f}% vs regex); {cap:.0f}% of the "
            f"regex→oracle headroom ({rgx:.3f}→{lr:.3f}→oracle {orc:.3f})."
        )
    report = "\n".join(lines)
    (PRIOR / "REPORT_routing.md").write_text(report, encoding="utf-8")
    log("REPORT_routing.md written")
    print("\n===FINAL===\n" + report, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama classifier tag, incl. :cloud (e.g. qwen3-vl:235b-cloud)",
    )
    ap.add_argument(
        "--prompt",
        default="classify_query",
        help="Classifier prompt name (e.g. classify_query_v2 — evidence-location)",
    )
    args = ap.parse_args()
    asyncio.run(main(args.model, args.prompt))
