"""Fetch MMDocIR: annotations plus the page images for a size-capped doc subset.

MMDocIR (arXiv 2501.08828) publishes 1,658 questions over 313 documents with
*both* page-level and layout-level (bbox) evidence labels. The layout labels are
what MMLongBench-Doc lacks: they say where on the page the answer lives, not just
which page.

The full corpus is ~20k pages, which is ~5 GB of ColQwen2 multivectors and hours
of GPU time on an 8 GB card. `--page-cap` drops the long tail of huge documents
(one has 843 pages); the default 60 keeps 218 docs / ~4.8k pages / ~1.15k
questions, which is an order of magnitude more queries than the committed
MMLongBench subset (n=107) at ~1.2 GB of index.

Page images ship inside `MMDocIR_pages.parquet` as JPEG bytes, so no PDF render
pass is needed. Pages are written straight out in the layout
`src.ingestion.visual.render_pages` uses (`<pages-dir>/<paper_id>/<paper_id>_p<N>`),
1-based page numbers, so `scripts.build_visual_index --pages-only` can index them.

Usage:
    .venv/Scripts/python.exe -m scripts.fetch_mmdocir \\
        --page-cap 60 \\
        --out data/mmdocir \\
        [--limit-docs 5]      # cap to N docs (alphabetical) for smoke testing

Outputs:
    data/mmdocir/annotations.jsonl      the 313-doc annotation file, verbatim
    data/mmdocir/manifest.json          the selected docs and their page counts
    data/mmdocir/pages/<paper_id>/...   page JPEGs for the selected docs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import src  # noqa: F401  -- loads .env

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HF_REPO_ID = "MMDocIR/MMDocIR_Evaluation_Dataset"
ANNOTATIONS_FILE = "MMDocIR_annotations.jsonl"
PAGES_FILE = "MMDocIR_pages.parquet"

# Keeps `<pages-dir>/<paper_id>/<paper_id>_p<N>.jpg` under the Windows 260-char
# path limit for the deepest doc name in the set.
_MAX_PAPER_ID_LEN = 48


def _paper_id(doc_name: str) -> str:
    """`PH_2016.06.08_Economy-Final.pdf` -> `PH_2016.06.08_Economy-Final`.

    Matches the repo's paper_id convention (PDF stem), which the golden set,
    the page chunk-ids and the visual index all key on.

    MMDocIR ships some doc names near 100 characters, and pages land at
    ``<pages-dir>/<paper_id>/<paper_id>_p<N>.jpg``, so the id appears twice, which
    overruns the 260-character Windows path limit. Long stems are therefore
    truncated and disambiguated with a digest of the full name.
    """
    stem = Path(doc_name).stem
    if len(stem) <= _MAX_PAPER_ID_LEN:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:6]
    return f"{stem[: _MAX_PAPER_ID_LEN - 7]}-{digest}"


def select_docs(
    rows: list[dict[str, Any]], page_cap: int, limit_docs: int | None
) -> list[dict[str, Any]]:
    """Docs with at most `page_cap` pages, alphabetical, optionally truncated."""
    kept = [r for r in rows if (r["page_indices"][1] - r["page_indices"][0]) <= page_cap]
    kept.sort(key=lambda r: str(r["doc_name"]))
    if limit_docs is not None:
        kept = kept[:limit_docs]
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/mmdocir"))
    parser.add_argument("--page-cap", type=int, default=60)
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed; run `uv sync` first.") from None

    args.out.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from HuggingFace ({HF_REPO_ID})...")
    ann_path = hf_hub_download(HF_REPO_ID, ANNOTATIONS_FILE, repo_type="dataset")
    shutil.copyfile(ann_path, args.out / "annotations.jsonl")
    with open(ann_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    print(f"  {len(rows)} docs, {sum(len(r['questions']) for r in rows)} questions")

    selected = select_docs(rows, args.page_cap, args.limit_docs)
    n_pages = sum(r["page_indices"][1] - r["page_indices"][0] for r in selected)
    n_queries = sum(len(r["questions"]) for r in selected)
    print(
        f"Selected {len(selected)} docs at --page-cap {args.page_cap}: "
        f"{n_pages} pages, {n_queries} questions"
    )

    # `page_indices` is a half-open range of *row positions* in the pages
    # parquet, explicitly NOT passage_ids (dataset README). passage_id is a
    # string identifier and must not be used to join. So the scan below counts
    # rows. Map every wanted row position to its (paper_id, 1-based page number).
    wanted: dict[int, tuple[str, int]] = {}
    for row in selected:
        paper_id = _paper_id(row["doc_name"])
        start, end = row["page_indices"]
        for offset, row_position in enumerate(range(start, end)):
            wanted[row_position] = (paper_id, offset + 1)

    print(f"Downloading {PAGES_FILE} (~1.6 GB, cached after the first run)...")
    pages_path = hf_hub_download(HF_REPO_ID, PAGES_FILE, repo_type="dataset")

    print("Writing page images...")
    written = 0
    row_position = 0
    parquet = pq.ParquetFile(pages_path)  # type: ignore[no-untyped-call]
    for batch in parquet.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=64, columns=["image_binary"]
    ):
        for image_binary in batch.column("image_binary").to_pylist():
            target = wanted.get(row_position)
            row_position += 1
            if target is None:
                continue
            paper_id, page_no = target
            paper_dir = pages_dir / paper_id
            paper_dir.mkdir(parents=True, exist_ok=True)
            out_path = paper_dir / f"{paper_id}_p{page_no}.jpg".replace(":", "_")
            if out_path.exists() and out_path.stat().st_size > 0:
                written += 1
                continue
            out_path.write_bytes(image_binary)
            written += 1
        if written and written % 1000 < 64:
            print(f"  {written}/{len(wanted)} pages")

    manifest = {
        "source": HF_REPO_ID,
        "page_cap": args.page_cap,
        "n_docs": len(selected),
        "n_pages_expected": len(wanted),
        "n_pages_written": written,
        "n_questions": n_queries,
        "docs": [
            {
                "doc_name": row["doc_name"],
                "paper_id": _paper_id(row["doc_name"]),
                "domain": row["domain"],
                "n_pages": row["page_indices"][1] - row["page_indices"][0],
                "n_questions": len(row["questions"]),
            }
            for row in selected
        ],
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {written}/{len(wanted)} pages to {pages_dir}")
    print(f"Manifest: {manifest_path}")
    if written < len(wanted):
        print("  WARNING: some selected pages were missing from the parquet.")


if __name__ == "__main__":
    main()
