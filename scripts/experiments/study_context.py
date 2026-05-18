"""Context-neighbourhood iteration harness (ADR 0016).

Tests whether expanding a retrieved chunk to its neighbourhood (page
neighbours + the figures/tables its text references) yields better
*answers* — the honest metric: answer-correctness vs the golden's
`expected_facts`, judged by an LLM, in the realistic **multi-doc**
setting (no paper filter — the hard case the user endorsed). Text
pipeline only (figures/tables are chunks in the text index; no ColQwen2,
so this is fast and keyless apart from Ollama gen+judge).

Decisive screen over the SAME PipelineRetriever via ContextExpansionRetriever:
  baseline (passthrough) vs +both (window+links), stratified ~10/bucket.
  A "win" is +both beating baseline by >= +0.03 mean correctness; the
  +window/+links ablation runs only if it wins. No reranker — it pinned the
  GPU and forced Ollama to CPU; it is a shared base so the delta is unbiased.

Sanity gate: baseline on a few queries must be non-zero, else ABORT
(gen/judge wiring broken) before the full run.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
import time
from pathlib import Path

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.ingestion.pipeline import ingest_paper
from src.llm.ollama_chat import OllamaChatClient
from src.llm.protocol import Message
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.generate import Generator
from src.rag.retrievers.context_expansion import ContextExpansionRetriever
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper, Query

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden" / "robust-v1.yaml"
ARXIV = ROOT / "data" / "papers"
MMLB = ROOT / "data" / "mmlongbench" / "documents"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
LLM = "gemma3:4b"
_BUCKET = re.compile(r"bucket=(\w+)")
PER_BUCKET = 10  # stratified screen: queries per answerable bucket
# Decisive screen: baseline vs the strongest arm only. The +window/+links
# ablation runs only if +both wins (ADR 0016).
ARMS = {  # name -> (window, link_artifacts)
    "baseline": (0, False),
    "+both": (2, True),
}
OUT = ROOT / "data" / "eval" / "runs" / f"context-study-{time.strftime('%Y%m%d-%H%M%S')}"
LOG = OUT / "driver.log"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _pdf_for(pid: str) -> Path | None:
    a, m = ARXIV / f"{pid}.pdf", MMLB / f"{pid}.pdf"
    return a if a.exists() else (m if m.exists() else None)


def _bucket(note: str | None) -> str:
    mm = _BUCKET.search(note or "")
    return mm.group(1) if mm else "text"


async def _judge(client: OllamaChatClient, q: str, ans: str, facts: list[str]) -> float:
    prompt = (
        "You grade whether an ANSWER correctly conveys the EXPECTED fact(s).\n"
        f"QUESTION: {q}\nEXPECTED: {' | '.join(facts)}\nANSWER: {ans}\n\n"
        "Reply with exactly one word: yes (answer states the expected fact "
        "correctly), partial (partially correct or incomplete), or no."
    )
    try:
        resp = await client.chat(
            messages=[Message(role="user", content=prompt)],
            model=LLM, temperature=0.0, images=None,
        )
    except Exception as e:
        log(f"  judge error ({type(e).__name__}) — scoring 0")
        return 0.0
    head = (resp.text or "").strip().lower()
    if head.startswith("yes"):
        return 1.0
    if head.startswith("partial"):
        return 0.5
    return 0.0


async def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    log("context-expansion study start")

    gs = load_golden_set(GOLDEN)
    answerable = [q for q in gs.queries if q.category != "out_of_corpus" and q.expected_facts]
    buckets = {q.query_id: _bucket(q.note) for q in answerable}
    _by_b: dict[str, list] = {}
    for q in sorted(answerable, key=lambda x: x.query_id):
        _by_b.setdefault(buckets[q.query_id], []).append(q)
    answerable = [q for qs in _by_b.values() for q in qs[:PER_BUCKET]]
    _dist = {b: min(len(v), PER_BUCKET) for b, v in _by_b.items()}
    log(f"stratified screen: {len(answerable)} queries {_dist}")
    pids = sorted({q.paper_id for q in answerable})
    pdfs = {p: _pdf_for(p) for p in pids}
    missing = [p for p, v in pdfs.items() if v is None]
    if missing:
        log(f"ABORT: missing PDFs {missing}")
        return
    log(f"{len(answerable)} answerable queries over {len(pids)} docs; ingesting")

    embedder = OllamaBgeEmbedder(base_url=OLLAMA)
    vs = QdrantVectorStore(url=QDRANT, collection_name="context_study", dim=embedder.dim)
    await vs.ensure_collection()
    bm25 = Bm25Index()
    chunks_by_id = {}
    for pid, pdf in pdfs.items():
        ing = await ingest_paper(
            paper=Paper(paper_id=pid, title=pid, pdf_path=pdf),
            embedder=embedder, vectorstore=vs, bm25=bm25,
            contextualizer_llm=None, contextualizer_model=None,
            contextualizer_concurrency=1,
            extract_figures_enabled=True, extract_tables_enabled=True,
            vlm_captioner=None,
        )
        for c in ing.chunks:
            chunks_by_id[c.chunk_id] = c
        log(f"  ingested {pid}: {ing.chunk_count}")
    # No cross-encoder reranker: it pins ~2 GB GPU and forces Ollama's
    # gemma3:4b onto CPU (~20x slower — root cause of the killed run). It is
    # a SHARED base across both arms, so dropping it does not bias the
    # baseline-vs-+both delta this study measures (ADR 0016).
    pipeline = PipelineRetriever(
        embedder=embedder, vectorstore=vs, bm25=bm25,
        chunks_by_id=chunks_by_id,
    )
    # gemma3:4b defaults to a 4096-token window; the generator packs up to
    # ~8000 tokens of context (and +both ~doubles it), so the default
    # truncates and garbles the prompt for BOTH arms — invalidating the
    # comparison (ADR 0016). 16384 holds full context+question+answer
    # untruncated so the only variable is the expansion content.
    client = OllamaChatClient(base_url=OLLAMA, num_ctx=16384)
    gen = Generator(llm=client, prompt=load_prompt_by_name("answer"), model=LLM)

    async def run_arm(name: str, queries: list) -> dict[str, list[float]]:
        window, link = ARMS[name]
        retr = ContextExpansionRetriever(
            base=pipeline, chunks_by_id=chunks_by_id,
            window=window, link_artifacts=link,
        )
        by_bucket: dict[str, list[float]] = {}
        for i, q in enumerate(queries):
            try:
                results = await retr.retrieve(Query(text=q.text, top_k=10))
                ans = await gen.answer(q.text, results)
                s = await _judge(client, q.text, ans.text, q.expected_facts)
            except Exception as e:
                log(f"  {name} q{i} error ({type(e).__name__}) — 0")
                s = 0.0
            by_bucket.setdefault(buckets[q.query_id], []).append(s)
            if (i + 1) % 20 == 0:
                log(f"  {name}: {i + 1}/{len(queries)}")
        return by_bucket

    # SANITY: baseline on first 8 must not be all-zero.
    probe = await run_arm("baseline", answerable[:8])
    pv = [x for v in probe.values() for x in v]
    log(f"SANITY baseline(8): mean={sum(pv) / len(pv):.3f}")
    if sum(pv) == 0:
        log("ABORT: baseline all-zero — gen/judge wiring broken.")
        return
    log("SANITY OK — running all arms.")

    import json

    results = {}
    for name in ARMS:
        results[name] = await run_arm(name, answerable)
        (OUT / "partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"done {name} (partial.json written)")

    bkts = ["text", "figure", "table", "mixed"]
    lines = ["# Context-expansion study — answer-correctness vs expected_facts\n",
             "robust-v1, multi-doc, no rerank (shared base — delta unbiased), "
             "gemma3:4b gen+judge. Stratified screen (ADR 0016).\n",
             "| arm | overall | " + " | ".join(bkts) + " | Δ vs baseline |",
             "|---|---|" + "---|" * (len(bkts) + 1)]
    overall = {}
    for name in ARMS:
        bb = results[name]
        allv = [x for v in bb.values() for x in v]
        o = sum(allv) / len(allv) if allv else 0.0
        overall[name] = o
        cells = []
        for b in bkts:
            v = bb.get(b, [])
            cells.append(f"{sum(v) / len(v):.3f}" if v else "—")
        d = "—" if name == "baseline" else f"{o - overall['baseline']:+.3f}"
        lines.append(f"| {name} | **{o:.4f}** | " + " | ".join(cells) + f" | {d} |")
    best = max((n for n in ARMS if n != "baseline"), key=lambda n: overall[n])
    margin = overall[best] - overall["baseline"]
    verdict = (
        f"WIN: **{best}** beats baseline by {margin:+.4f} (>= +0.03 bar)."
        if margin >= 0.03
        else f"NO WIN: best arm ({best}) only {margin:+.4f} vs baseline "
        "(< +0.03) — context-expansion is not a real lever here. Honest null."
    )
    lines += ["", verdict]
    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    log("REPORT.md written")
    print("\n===REPORT===\n" + report, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
