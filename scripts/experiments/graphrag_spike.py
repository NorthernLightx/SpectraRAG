"""GraphRAG kill-spike (ADR 0018 M2) — decisive, cheap, fully local.

Does GraphRAG-global beat plain BM25-RAG on *global synthesis* questions —
the only class it should win, since hybrid already saturates factoid lookup
(ADR 0015/0016)? On 2-3 papers, ~hundreds of local LLM calls, no Qdrant, no
Docker. Control arm is the in-process BM25 retriever (same chunks, same
model, same answer prompt — only retrieval differs). Throwaway: experiments
tier, exempt from gates. Read the side-by-side, decide continue/kill.

Doubles as the graph-axis of the ingestion scorecard: writes graph-quality
metrics (bib-flag rate, entities/chunk, isolates, community shape) so graph
ingestion quality is transparent *before* any 20-paper build.

    uv run python -m scripts.experiments.graphrag_spike --model gemma3:4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

from src.graph import build_graph, detect_communities, summarize_communities
from src.ingestion.chunking import chunk_pages
from src.ingestion.graph_extract import extract_graph
from src.ingestion.pdf import extract_pages
from src.llm.ollama_chat import OllamaChatClient
from src.llm.protocol import Message
from src.rag.bm25 import Bm25Index
from src.types import Chunk, ChunkExtraction, CommunityReport

# The corpus is full of ∥ ∑ θ etc.; the Windows cp1252 console crashed the
# *previous* run on a print AFTER the 34-min LLM work. Never again.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Genuinely cross-cutting — the class single-passage retrieval should struggle
# on and graph community summaries should help. No paper-specific assumptions.
_GLOBAL_QUERIES = [
    "What problem domains do these papers address, and what do they share?",
    "Which methods or techniques are introduced or compared across these papers?",
    "What evaluation metrics or benchmarks recur across these papers?",
    "What limitations or open problems do these papers acknowledge?",
    "What datasets are used across these papers?",
    "What are the main contributions claimed across these papers?",
    "How do these papers relate to scaling, efficiency, or optimization?",
    "What future-work directions are suggested across these papers?",
]

_ANSWER = (
    "Answer the question using ONLY the context. Be specific and concise. "
    "If the context does not support an answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
)


async def _answer(llm: OllamaChatClient, model: str, q: str, context: str) -> str:
    resp = await llm.chat(
        messages=[Message(role="user", content=_ANSWER.format(context=context[:8000], q=q))],
        model=model,
        temperature=0.0,
        max_tokens=400,
    )
    return resp.text.strip()


def _graph_metrics(extractions: list, graph: object, communities: list) -> dict[str, object]:
    import networkx as nx

    g: nx.Graph = graph  # type: ignore[assignment]
    n = len(extractions) or 1
    sizes = [len(c.entity_names) for c in communities]
    return {
        "chunks": len(extractions),
        "bib_flagged_pct": round(100 * sum(e.is_reference_list for e in extractions) / n, 1),
        "zero_extraction_pct": round(
            100 * sum(not e.entities and not e.is_reference_list for e in extractions) / n, 1
        ),
        "entities_per_chunk": round(sum(len(e.entities) for e in extractions) / n, 2),
        "relations_per_chunk": round(sum(len(e.relations) for e in extractions) / n, 2),
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "isolates_pct": round(
            100 * nx.number_of_isolates(g) / (g.number_of_nodes() or 1), 1
        ),
        "communities": len(communities),
        "singleton_community_pct": round(
            100 * sum(s <= 1 for s in sizes) / (len(sizes) or 1), 1
        ),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--papers",
        nargs="+",
        default=["2604.22753v1", "2604.28173v1", "2604.28192v1"],
        help="2-3 paper ids (diverse domains stress global synthesis)",
    )
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--limit", type=int, default=0, help="cap chunks (smoke wiring fast); 0 = all")
    ap.add_argument("--top-bm25", type=int, default=6)
    ap.add_argument("--top-reports", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("data/eval/ingestion/spike.md"))
    ap.add_argument("--cache", type=Path, default=Path("data/eval/ingestion/spike-cache.json"))
    ap.add_argument("--refresh", action="store_true", help="ignore cache, redo the LLM passes")
    args = ap.parse_args()

    # num_ctx bumped: gemma3:4b's 4096 default truncated prompts in ADR 0016 —
    # do not repeat that artifact in the experiment meant to avoid it.
    llm = OllamaChatClient(num_ctx=16384)

    chunks: list[Chunk] = []
    for pid in args.papers:
        chunks.extend(chunk_pages(extract_pages(pid, Path(f"data/papers/{pid}.pdf"))))
    if args.limit:
        chunks = chunks[: args.limit]

    # Re-chunking is cheap (no LLM); extraction + summarisation are the 34-min
    # cost. Cache only those so verdict iteration on the query loop is seconds.
    if args.cache.exists() and not args.refresh:
        cached = json.loads(args.cache.read_text(encoding="utf-8"))
        extractions = [ChunkExtraction.model_validate(e) for e in cached["extractions"]]
        reports = [CommunityReport.model_validate(r) for r in cached["reports"]]
        graph = build_graph(extractions)
        communities = detect_communities(graph)
        print(f"loaded cache: {len(extractions)} extractions, {len(reports)} reports")
    else:
        print(f"{len(chunks)} chunks from {len(args.papers)} papers; extracting graph...")
        extractions = await extract_graph(chunks, llm=llm, model=args.model)
        graph = build_graph(extractions)
        communities = detect_communities(graph)
        reports = await summarize_communities(graph, communities, llm=llm, model=args.model)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            json.dumps(
                {
                    "extractions": [e.model_dump() for e in extractions],
                    "reports": [r.model_dump() for r in reports],
                }
            ),
            encoding="utf-8",
        )
        print(f"cached {len(extractions)} extractions + {len(reports)} reports to {args.cache}")

    metrics = _graph_metrics(extractions, graph, communities)
    metrics["reports_emitted"] = len(reports)
    Path("data/eval/ingestion").mkdir(parents=True, exist_ok=True)
    Path("data/eval/ingestion/spike-graph.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("graph-ingestion metrics:", json.dumps(metrics, indent=2))

    bm25 = Bm25Index()
    bm25.add(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    report_blob = "\n".join(f"{r.title}: {r.summary}" for r in reports)

    md = ["# GraphRAG kill-spike — side-by-side", "", json.dumps(metrics), ""]
    for q in _GLOBAL_QUERIES:
        qw = {w for w in q.lower().split() if len(w) > 3}
        ranked = sorted(
            reports,
            key=lambda r: len(qw & set((r.title + " " + r.summary).lower().split())),
            reverse=True,
        )
        graph_ctx = "\n".join(f"{r.title}: {r.summary}" for r in ranked[: args.top_reports]) or report_blob
        hits = bm25.search(q, args.top_bm25)
        bm25_ctx = "\n\n".join(by_id[h.chunk_id].text for h in hits)
        graph_ans = await _answer(llm, args.model, q, graph_ctx)
        bm25_ans = await _answer(llm, args.model, q, bm25_ctx)
        md += [
            f"## {q}",
            f"**GraphRAG-global:** {graph_ans}",
            "",
            f"**BM25-RAG:** {bm25_ans}",
            "",
            "---",
        ]
        print(f"\n=== {q}\n[GRAPH] {graph_ans[:280]}\n[BM25 ] {bm25_ans[:280]}")

    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out} — read the side-by-side, decide continue/kill.")
    lens = [len(c.text) for c in chunks]
    print(f"(corpus: {len(chunks)} chunks, mean {statistics.fmean(lens):.0f} chars)")


if __name__ == "__main__":
    asyncio.run(main())
