"""Ingest the full MMLongBench doc set into a new `mmlb_v2` collection.

Extends the eval corpus from 20 docs (routing_study) to all 134 locally-available
docs, so the mmlongbench-v2 golden (821 in-corpus queries) is end-to-end scorable
and future lever A/Bs have ~7.5x the statistical power.

Text leg only (docling chunker + bge-m3 + BM25 via Qdrant payload) — the same
pipeline the committed text baseline uses. The OOM-prone ColQwen2 visual index is
a separate build, deferred. Non-destructive: writes a NEW collection, leaving
`routing_study` (the committed baseline corpus) untouched.

Resumable: skips any paper already present in the target collection, so a restart
after an OOM / crash continues where it stopped. Per-doc error isolation — one bad
PDF doesn't abort the batch.

Usage:
    .venv/Scripts/python.exe -m scripts.ingest_mmlb_v2 \
        --collection mmlb_v2 --limit 3        # validate on 3 docs first
    .venv/Scripts/python.exe -m scripts.ingest_mmlb_v2 --collection mmlb_v2  # full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from qdrant_client import models as qm

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.ingestion.pipeline import ingest_paper
from src.observability.logging import configure_logging
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


async def _present(vs: QdrantVectorStore, paper_id: str) -> bool:
    """True if the collection already has chunks for this paper (resume support)."""
    res = await vs._client.count(
        collection_name=vs._collection,
        count_filter=qm.Filter(
            must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))]
        ),
        exact=True,
    )
    return res.count > 0


async def run(args: argparse.Namespace) -> int:
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    want = sorted({q["paper_id"] for q in golden["queries"]})
    pdfs = {p.stem: p for p in args.docs_dir.glob("*.pdf")}
    targets = [(pid, pdfs[pid]) for pid in want if pid in pdfs]
    missing = [pid for pid in want if pid not in pdfs]
    if args.limit:
        targets = targets[: args.limit]
    print(
        f"v2 golden references {len(want)} papers; {len(targets)} have a local PDF"
        f"{f' (limited to {args.limit})' if args.limit else ''}; {len(missing)} PDFs missing"
    )

    embedder = OllamaBgeEmbedder(base_url=args.ollama)
    vs = QdrantVectorStore(url=args.qdrant, collection_name=args.collection, dim=embedder.dim)
    await vs.ensure_collection()

    done: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    for i, (pid, path) in enumerate(targets, 1):
        if await _present(vs, pid):
            skipped.append(pid)
            print(f"[{i}/{len(targets)}] skip (present): {pid}")
            continue
        try:
            paper = Paper(paper_id=pid, title=pid, pdf_path=path)
            result = await ingest_paper(
                paper=paper, embedder=embedder, vectorstore=vs, bm25=Bm25Index()
            )
            done.append(pid)
            print(f"[{i}/{len(targets)}] ingested {pid}: {result.chunk_count} chunks")
        except Exception as exc:  # one bad PDF must not abort the batch
            failed.append({"paper_id": pid, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i}/{len(targets)}] FAILED {pid}: {type(exc).__name__}: {exc}")
        if args.progress:
            args.progress.write_text(
                json.dumps(
                    {
                        "collection": args.collection,
                        "done": done,
                        "skipped": skipped,
                        "failed": failed,
                        "missing_pdf": missing,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    total = await vs.count()
    print(
        f"\nDone. ingested={len(done)} skipped={len(skipped)} failed={len(failed)} "
        f"| collection '{args.collection}' now has {total} points"
    )
    if failed:
        print("  failures:", [f["paper_id"] for f in failed])
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v2.yaml"))
    ap.add_argument("--docs-dir", type=Path, default=Path("data/mmlongbench/documents"))
    ap.add_argument("--collection", default="mmlb_v2")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument(
        "--limit", type=int, default=0, help="ingest only the first N target docs (0=all)"
    )
    ap.add_argument(
        "--progress", type=Path, default=Path("data/eval/runs/ingest_mmlb_v2_progress.json")
    )
    args = ap.parse_args()

    configure_logging(level="WARNING", env="local", log_file=Path("logs") / "ingest_mmlb_v2.log")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
