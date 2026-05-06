"""Render every PDF in --pdf-dir to PNGs under --out-dir.

Used at Docker build time to bake the visual leg's page images into the
runtime image: the deployed `/answer` UI sends those page URLs to OpenRouter
as image content blocks, so a vision-capable model (gpt-4o, claude, qwen3-vl)
can read pixels even though the deploy has no GPU for ColQwen2 retrieval.
Idempotent — `render_pages()` skips files that already exist.

Usage:

    .venv/Scripts/python.exe -m scripts.render_pages \
        --pdf-dir data/papers --out-dir data/pages --dpi 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.visual import render_pages
from src.observability.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/pages"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    configure_logging(level="INFO", env="local", log_file=None)
    log = get_logger("scripts.render_pages")

    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No .pdf files found in {args.pdf_dir}")

    total_pages = 0
    for pdf_path in pdf_paths:
        rendered = render_pages(pdf_path.stem, pdf_path, out_dir=args.out_dir, dpi=args.dpi)
        total_pages += len(rendered)
        print(f"  {pdf_path.name}: {len(rendered)} pages")

    print(f"\nRendered {total_pages} pages across {len(pdf_paths)} PDFs to {args.out_dir}")
    log.info(
        "render_pages.done",
        papers=len(pdf_paths),
        pages=total_pages,
        out_dir=str(args.out_dir),
    )


if __name__ == "__main__":
    main()
