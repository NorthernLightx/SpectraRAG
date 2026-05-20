"""Render the smallest N figure-kind chunks with their bbox overlaid on the
page PNG, so we can characterize what Docling's layout model is labelling
as "picture" at small sizes.

Output: data/tiny_figures_inspect/<paper>__<page>__<chunk>.png — page crop
around the bbox with the bbox drawn in red. Plus a single contact-sheet
montage for quick comparison.

Run:
  python scripts/experiments/inspect_tiny_figures.py --collection eval_docling_mm --n 30
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

QDRANT = "http://localhost:6333"
PAGES_DIR = Path("data/pages")
OUT_DIR = Path("data/tiny_figures_inspect")
PAGE_DPI = 150  # the render DPI used by the ingestion pipeline


def fetch_figure_chunks(collection: str) -> list[dict]:
    """Scroll all figure-kind chunks from the collection."""
    payload = {
        "limit": 1000,
        "with_payload": True,
        "with_vector": False,
        "filter": {"must": [{"key": "metadata.kind", "match": {"value": "figure"}}]},
    }
    req = urllib.request.Request(
        f"{QDRANT}/collections/{collection}/points/scroll",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read())
    return body["result"]["points"]


def crop_with_bbox(
    page_png: Path, bbox_pts: list[float], pad_pts: float = 60.0
) -> Image.Image:
    """Open the page PNG, convert bbox from PDF points to image pixels using the
    rendered DPI, crop with padding, and draw the original bbox in red."""
    img = Image.open(page_png).convert("RGB")
    iw, ih = img.size
    # pageW_pts = iw * 72 / DPI -> px_per_pt = DPI / 72
    px = PAGE_DPI / 72.0
    x0, y0, x1, y1 = (v * px for v in bbox_pts)
    # crop bounds with padding
    cx0 = max(0, int(x0 - pad_pts * px))
    cy0 = max(0, int(y0 - pad_pts * px))
    cx1 = min(iw, int(x1 + pad_pts * px))
    cy1 = min(ih, int(y1 + pad_pts * px))
    crop = img.crop((cx0, cy0, cx1, cy1)).copy()
    d = ImageDraw.Draw(crop)
    # bbox coords in crop space
    box = (int(x0 - cx0), int(y0 - cy0), int(x1 - cx0), int(y1 - cy0))
    d.rectangle(box, outline=(255, 30, 30), width=3)
    return crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="eval_docling_mm")
    ap.add_argument("--n", type=int, default=30, help="How many of the smallest to render")
    ap.add_argument("--pad-pts", type=float, default=60.0)
    args = ap.parse_args()

    pts = fetch_figure_chunks(args.collection)
    figs = []
    for p in pts:
        pl = p["payload"]
        bb = pl.get("metadata", {}).get("bbox")
        if not bb:
            continue
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        figs.append(
            {
                "chunk_id": pl["chunk_id"],
                "paper_id": pl["paper_id"],
                "pages": pl.get("page_numbers", []),
                "bbox": bb,
                "area": w * h,
                "size": (w, h),
            }
        )
    figs.sort(key=lambda f: f["area"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = figs[: args.n]
    print(f"figure chunks total: {len(figs)}; rendering smallest {len(selected)}")

    saved: list[tuple[str, Path]] = []
    for f in selected:
        page_no = f["pages"][0] if f["pages"] else None
        if page_no is None:
            continue
        # page PNG: data/pages/<paper>/<paper>_p<N>.png
        page_png = PAGES_DIR / f["paper_id"] / f"{f['paper_id']}_p{page_no}.png"
        if not page_png.exists():
            print(f"  missing page png: {page_png}")
            continue
        try:
            crop = crop_with_bbox(page_png, f["bbox"], pad_pts=args.pad_pts)
        except Exception as e:
            print(f"  fail {f['chunk_id']}: {e}")
            continue
        w, h = f["size"]
        label = f"{f['chunk_id']}  {w:.0f}x{h:.0f}pt  area={f['area']:.0f}"
        # paint the label above the crop
        label_h = 26
        canvas = Image.new("RGB", (max(crop.width, 400), crop.height + label_h), (20, 20, 24))
        canvas.paste(crop, (0, label_h))
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        d.text((6, 4), label, fill=(220, 220, 220), font=font)
        out_path = OUT_DIR / f"{f['chunk_id'].replace('::', '__')}.png"
        canvas.save(out_path)
        saved.append((label, out_path))
        print(f"  {label} -> {out_path.name}")

    # Build a contact-sheet montage: 4 cols
    if saved:
        cols = 4
        rows = (len(saved) + cols - 1) // cols
        thumbs = [Image.open(p).convert("RGB") for _, p in saved]
        tw = 360
        # uniform thumb width preserving aspect ratio
        rescaled = []
        for t in thumbs:
            ratio = tw / t.width
            rescaled.append(t.resize((tw, int(t.height * ratio))))
        rh = max(t.height for t in rescaled)
        sheet = Image.new("RGB", (cols * tw, rows * rh), (10, 10, 12))
        for i, t in enumerate(rescaled):
            r, c = divmod(i, cols)
            sheet.paste(t, (c * tw, r * rh))
        sheet_path = OUT_DIR / "_contact_sheet.png"
        sheet.save(sheet_path)
        print(f"\ncontact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
