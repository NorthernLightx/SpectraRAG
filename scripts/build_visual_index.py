"""Build the persisted ColQwen2 page index into a Qdrant multivector collection.

Run once, offline, on a machine with a GPU (ADR 0028). It renders every PDF
page, embeds each page with ColQwen2, and upserts the page multivectors into
``--collection`` inside the same embedded Qdrant store the text corpus lives in.
The deploy then loads these vectors at startup instead of re-encoding pages,
which is what makes the visual leg viable on a CPU-only Cloud Run box.

Encoding reuses ``build_visual_retriever``, the exact path the offline eval
scores, so the persisted vectors are identical to the eval's in-memory ones.
That is what lets the deployed router reproduce the eval's recall number.

Memory: this loads every page embedding into memory at once (same profile as
the in-memory eval leg). Run it on the GPU box with no other GPU work in flight
(the 8 GB card OOMs if Ollama or a desktop compositor is holding VRAM).

Usage:

    .venv/Scripts/python.exe -m scripts.build_visual_index \\
        --pdf-dir data/papers \\
        --qdrant path:./qdrant_local \\
        --collection rag_corpus_visual \\
        --pages-dir data/pages

Idempotent: refuses to rebuild a non-empty collection unless ``--force``
(which drops only the visual collection, leaving the text collection intact).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
from structlog.stdlib import BoundLogger

from src.observability.logging import configure_logging, get_logger
from src.rag.retrievers.visual import build_visual_retriever
from src.rag.visual_store import QdrantVisualStore

_UPSERT_BATCH = 32
_PAGE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _scan_pages_dir(pages_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Read an already-rendered page tree into ``paper_id -> [(page_no, path)]``.

    Layout is the one ``src.ingestion.visual.render_pages`` writes:
    ``<pages-dir>/<paper_id>/<paper_id>_p<N>.<ext>``, 1-based page numbers.
    """
    pages_by_paper: dict[str, list[tuple[int, Path]]] = {}
    for paper_dir in sorted(p for p in pages_dir.iterdir() if p.is_dir()):
        paper_id = paper_dir.name
        pages: list[tuple[int, Path]] = []
        for image_path in paper_dir.iterdir():
            if image_path.suffix.lower() not in _PAGE_IMAGE_SUFFIXES:
                continue
            _, _, page_part = image_path.stem.rpartition("_p")
            if not page_part.isdigit():
                continue
            pages.append((int(page_part), image_path))
        if pages:
            pages_by_paper[paper_id] = sorted(pages)
    return pages_by_paper


async def _main(
    *,
    pdf_dir: Path,
    pdfs: list[Path] | None,
    qdrant_url: str,
    collection: str,
    corpus_collection: str,
    pages_dir: Path,
    visual_model: str,
    device: str,
    dpi: int,
    force: bool,
    pages_only: bool = False,
) -> None:
    log = get_logger("scripts.build_visual_index")
    if pages_only:
        await _main_pages_only(
            qdrant_url=qdrant_url,
            collection=collection,
            pages_dir=pages_dir,
            visual_model=visual_model,
            device=device,
            force=force,
            log=log,
        )
        return
    pdf_paths = sorted(pdfs) if pdfs else sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No .pdf files found in {pdf_dir}")

    # Scope to the served text corpus: the visual index must cover exactly the
    # papers `corpus_collection` holds. Globbing data/papers otherwise pulls in
    # the ingestion-robustness fixtures (het-apollo17's 339 pages, het-*-slides,
    # het-hal-fr) that are not served. They would bloat the baked index and
    # surface visual pages with no text counterpart. Read + close before the
    # encode so the embedded path-mode lock is free for the upsert step (local
    # mode rejects two live clients on one path).
    from src.rag.vectorstore import QdrantVectorStore

    text_store = QdrantVectorStore(url=qdrant_url, collection_name=corpus_collection, dim=1024)
    try:
        corpus_papers = {c.paper_id for c in await text_store.scroll_chunks()}
    finally:
        await text_store._client.close()
    if corpus_papers:
        skipped = sorted(p.stem for p in pdf_paths if p.stem not in corpus_papers)
        pdf_paths = [p for p in pdf_paths if p.stem in corpus_papers]
        if skipped:
            print(
                f"Skipping {len(skipped)} PDFs not in {corpus_collection!r}: {', '.join(skipped)}"
            )
    else:
        print(f"WARNING: {corpus_collection!r} is empty; encoding all globbed PDFs unscoped")
    if not pdf_paths:
        raise SystemExit(f"No PDFs left after scoping to {corpus_collection!r}.")
    print(f"Encoding {len(pdf_paths)} papers (scoped to {corpus_collection!r})")

    if not await _prepare_collection(qdrant_url, collection, force, log):
        return

    # Render every page to PNG (idempotent), then encode via the same path the
    # eval uses so the persisted vectors match it.
    from src.ingestion.visual import render_pages

    pages_by_paper: dict[str, list[tuple[int, Path]]] = {}
    for pdf_path in pdf_paths:
        paper_id = pdf_path.stem
        rendered = render_pages(paper_id, pdf_path, out_dir=pages_dir, dpi=dpi)
        pages_by_paper[paper_id] = [(r.page_number, r.image_path) for r in rendered]
        print(f"  {paper_id}: {len(rendered)} pages")

    await _embed_and_persist(
        pages_by_paper,
        qdrant_url=qdrant_url,
        collection=collection,
        visual_model=visual_model,
        device=device,
        log=log,
    )


async def _prepare_collection(
    qdrant_url: str, collection: str, force: bool, log: BoundLogger
) -> bool:
    """Idempotency / ``--force`` gate. Returns False when the build should skip.

    Closes the client before returning either way: embedded path-mode allows one
    live client per on-disk path, and the encode + upsert step reopens it.
    """
    store = QdrantVisualStore(url=qdrant_url, collection_name=collection)
    try:
        existing = await store.count()
        if existing > 0 and not force:
            print(
                f"Collection {collection!r} already has {existing} pages; skipping (--force to rebuild)."
            )
            log.info("build_visual.skip", collection=collection, existing=existing)
            return False
        if existing > 0 and force:
            print(f"--force: dropping {existing} pages in {collection!r}")
            await store.delete_collection()
    finally:
        await store.close()
    return True


async def _main_pages_only(
    *,
    qdrant_url: str,
    collection: str,
    pages_dir: Path,
    visual_model: str,
    device: str,
    force: bool,
    log: BoundLogger,
) -> None:
    """Index an already-rendered page tree: no PDFs, no text-corpus scoping.

    The benchmark corpora ship page images rather than renderable PDFs
    (``scripts.fetch_mmdocir``), and a visual-leg recall run does not need the
    text collection that the PDF path scopes against.
    """
    pages_by_paper = _scan_pages_dir(pages_dir)
    if not pages_by_paper:
        raise SystemExit(f"No page images found under {pages_dir}")
    n_pages = sum(len(v) for v in pages_by_paper.values())
    print(f"Found {n_pages} pages across {len(pages_by_paper)} papers in {pages_dir}")

    if not await _prepare_collection(qdrant_url, collection, force, log):
        return

    await _embed_and_persist(
        pages_by_paper,
        qdrant_url=qdrant_url,
        collection=collection,
        visual_model=visual_model,
        device=device,
        log=log,
    )


async def _embed_and_persist(
    pages_by_paper: dict[str, list[tuple[int, Path]]],
    *,
    qdrant_url: str,
    collection: str,
    visual_model: str,
    device: str,
    log: BoundLogger,
) -> None:
    """Encode every page and upsert the multivectors into ``collection``."""
    print(f"Embedding pages with {visual_model} on {device} (the slow part)...")
    retriever = await build_visual_retriever(pages_by_paper, model_name=visual_model, device=device)
    page_embeds = retriever._page_embeds  # chunk_id -> [n_patches, dim] tensor
    page_meta = retriever._page_meta  # chunk_id -> (paper_id, page_no)
    if not page_embeds:
        raise SystemExit("No pages embedded; nothing to persist.")

    dim = int(next(iter(page_embeds.values())).shape[-1])
    chunk_ids = list(page_embeds)

    out_store = QdrantVisualStore(url=qdrant_url, collection_name=collection, dim=dim)
    try:
        await out_store.ensure_collection()
        # Convert one upsert batch at a time. A page is ~1k patches x 128 dims,
        # and a Python list of floats costs ~32 bytes per element, so
        # materializing a whole benchmark corpus (thousands of pages) at once
        # runs to tens of GB of host memory. Tensors are dropped as they go.
        for start in range(0, len(chunk_ids), _UPSERT_BATCH):
            batch: list[tuple[str, int, list[list[float]]]] = []
            for chunk_id in chunk_ids[start : start + _UPSERT_BATCH]:
                paper_id, page_no = page_meta[chunk_id]
                batch.append((paper_id, page_no, page_embeds[chunk_id].float().cpu().tolist()))
                del page_embeds[chunk_id]
            await out_store.upsert_pages(batch)
        final = await out_store.count()
        print(f"\nPersisted {final} page vectors (dim={dim}) into {collection!r}")
        log.info(
            "build_visual.done",
            collection=collection,
            papers=len(pages_by_paper),
            pages=final,
            dim=dim,
            model=visual_model,
        )
    finally:
        await out_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render + embed PDF pages into a Qdrant multivector collection."
    )
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/papers"))
    parser.add_argument(
        "--pdf",
        dest="pdfs",
        type=Path,
        action="append",
        help="Embed specific PDF file(s) instead of globbing --pdf-dir (repeatable).",
    )
    parser.add_argument("--qdrant", default="path:./qdrant_local")
    parser.add_argument("--collection", default="rag_corpus_visual")
    parser.add_argument(
        "--corpus-collection",
        default="rag_corpus",
        help="Text collection to scope the visual index to (encode only its papers).",
    )
    parser.add_argument("--pages-dir", type=Path, default=Path("data/pages"))
    parser.add_argument("--visual-model", default="vidore/colqwen2-v1.0")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for embedding. Defaults to cuda when available.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the collection is non-empty (drops only this collection).",
    )
    parser.add_argument(
        "--pages-only",
        action="store_true",
        help="Index the page images already under --pages-dir; skip PDFs and corpus scoping.",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", env="local", log_file=None)
    asyncio.run(
        _main(
            pdf_dir=args.pdf_dir,
            pdfs=args.pdfs,
            qdrant_url=args.qdrant,
            collection=args.collection,
            corpus_collection=args.corpus_collection,
            pages_dir=args.pages_dir,
            visual_model=args.visual_model,
            device=args.device,
            dpi=args.dpi,
            force=args.force,
            pages_only=args.pages_only,
        )
    )
