"""``spectrarag`` command-line entry point.

A thin convenience layer over the documented workflows: one command instead of
a chain of ``python -m scripts.*``. ``serve`` runs the API self-contained
(in-process bge-m3 + the committed embedded Qdrant snapshot, no Docker or Ollama
needed); ``ingest`` and ``fetch`` wrap the existing scripts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable


def _run_module(module: str, args: list[str]) -> int:
    """Run ``python -m <module> <args>`` in a subprocess; return its exit code."""
    return subprocess.run([sys.executable, "-m", module, *args], check=False).returncode


def _serve(ns: argparse.Namespace) -> int:
    # Self-contained defaults so `spectrarag serve` works after a clone with no
    # external services: the `cpu` profile (in-process sentence-transformers
    # bge-m3, no Ollama; the MiniLM reranker, CPU-feasible and non-inferior on
    # the eval sets per ADR 0012) and the committed embedded Qdrant snapshot (no
    # Qdrant server). setdefault only fills these when the user hasn't set them
    # via the shell or .env.
    os.environ.setdefault("RAG_PROFILE", "cpu")
    # `.env` (copied from `.env.example`) ships RAG_QDRANT_URL=http://localhost:6333,
    # the docker-compose default, and `src/__init__` loads it before this runs.
    # That would point the self-contained serve at a Qdrant server that isn't up,
    # so treat the docker default as "unset" and use the embedded snapshot; a
    # genuinely custom URL (a real server the user runs) is left untouched.
    _docker_qdrant = "http://localhost:6333"
    if os.environ.get("RAG_QDRANT_URL", _docker_qdrant) == _docker_qdrant:
        os.environ["RAG_QDRANT_URL"] = "path:./qdrant_local"
    if os.path.isdir("data/pages"):
        os.environ.setdefault("RAG_PAGES_DIR", "data/pages")

    print("Starting SpectraRAG. The first run loads the embedding + reranker models (~30-60s).")

    import uvicorn

    uvicorn.run("src.api.main:app", host=ns.host, port=ns.port, reload=ns.reload)
    return 0


def _ingest(ns: argparse.Namespace) -> int:
    args = ["--pdf-dir", ns.pdf_dir, "--collection", ns.collection]
    if ns.force:
        args.append("--force")
    return _run_module("scripts.bootstrap_corpus", args)


def _fetch(ns: argparse.Namespace) -> int:
    args = ["--out-dir", ns.out_dir]
    if ns.manifest:
        args += ["--manifest", ns.manifest]
    return _run_module("scripts.fetch_papers", args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectrarag",
        description="Multi-modal PDF RAG: serve the API, ingest PDFs, or fetch the demo corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve", help="Run the API self-contained (in-process bge-m3 + embedded Qdrant)."
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind address.")
    serve.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")
    serve.set_defaults(func=_serve)

    ingest = sub.add_parser(
        "ingest", help="Ingest a folder of PDFs into a collection (needs Ollama for bge-m3)."
    )
    ingest.add_argument("--pdf-dir", default="data/papers", help="Folder of PDFs to ingest.")
    ingest.add_argument("--collection", default="rag_corpus", help="Target Qdrant collection name.")
    ingest.add_argument(
        "--force", action="store_true", help="Re-ingest into a non-empty collection."
    )
    ingest.set_defaults(func=_ingest)

    fetch = sub.add_parser("fetch", help="Download demo-corpus PDFs from a manifest of arXiv IDs.")
    fetch.add_argument(
        "--manifest", default="data/curated_demo/papers.txt", help="Manifest of arXiv IDs to fetch."
    )
    fetch.add_argument(
        "--out-dir", default="data/papers", help="Where to save the downloaded PDFs."
    )
    fetch.set_defaults(func=_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = ns.func
    return func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
