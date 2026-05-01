"""Dump `chunk_id | pages | first 280 chars` per chunk for the given PDF.

Used when authoring or revising golden-set queries: lets the human reviewer
match a query intent to a concrete chunk_id without re-running the full
ingestion pipeline.

Run:
  uv run python -m scripts.dump_chunks \
      --pdf data/papers/2604.22753v1.pdf \
      --out chunks_dump.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.chunking import chunk_pages
from src.ingestion.pdf import extract_pages


def _format_line(chunk_id: str, pages: list[int], text: str) -> str:
    flat = text.replace("\n", " ")[:280]
    return f"{chunk_id} | pp={pages} | {flat}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("chunks_dump.txt"))
    parser.add_argument("--target-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=200)
    args = parser.parse_args()

    pages = extract_pages(args.pdf.stem, args.pdf)
    chunks = chunk_pages(pages, target_chars=args.target_chars, overlap_chars=args.overlap_chars)
    lines = [_format_line(c.chunk_id, c.page_numbers, c.text) for c in chunks]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
