"""Pre-render small WebP thumbnails for every gallery figure/table.

The web UI's figure thumbnails were CSS-crops of the *full* page PNG (~0.5 MB
each), so opening a paper with many figures pulled megabytes before anything
showed. This renders a small thumbnail per gallery item once, served statically
from the existing ``/pages`` mount, so the thumbnail becomes a plain ``<img>``.

Source per item:
  - figure: downscale the Docling crop already on disk (``data/figures/<paper>/
    <id>.png``; ``id`` is the chunk_id with ``:`` → ``_``).
  - table (no crop) or a figure whose crop is missing: crop the bbox region out
    of the page render (``data/pages/<paper>/<paper>_p<N>.png``). Pages render
    at 150 DPI, bbox is in PDF points (1/72"), so px = pt * 150/72.

Output: ``<pages_dir>/<paper>/thumbs/<id>.webp`` (committed with the page PNGs so
the self-contained prod image bundles them). Items are read from a running
server's ``/figures`` so the set matches exactly what the gallery shows
(decoration noise already filtered).

Run (server up): ``uv run python -m scripts.render_figure_thumbs``
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

_DPI = 150  # render_pages default; bbox points → px is pt * _DPI/72
_PT_TO_PX = _DPI / 72.0


def _safe_id(chunk_id: str) -> str:
    # Mirror docling_parser._safe_filename so the crop file / thumb name match.
    return chunk_id.replace(":", "_")


def _fetch_items(base_url: str) -> list[dict[str, object]]:
    with urllib.request.urlopen(f"{base_url}/figures?limit=1000", timeout=30) as resp:
        items: list[dict[str, object]] = json.load(resp)
        return items


def _thumb_from_crop(crop_path: Path, max_px: int) -> Image.Image:
    img = Image.open(crop_path).convert("RGB")
    img.thumbnail((max_px, max_px))
    return img


def _thumb_from_page(page_path: Path, bbox: list[float], max_px: int) -> Image.Image:
    img = Image.open(page_path).convert("RGB")
    x0, y0, x1, y1 = (v * _PT_TO_PX for v in bbox)
    # Clamp to the page so a slightly-oversized bbox can't error.
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    crop = img.crop((round(x0), round(y0), round(x1), round(y1)))
    crop.thumbnail((max_px, max_px))
    return crop


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Running server.")
    p.add_argument("--pages-dir", type=Path, default=Path("data/pages"))
    p.add_argument("--figures-dir", type=Path, default=Path("data/figures"))
    p.add_argument("--max-px", type=int, default=512, help="Longest thumbnail edge.")
    p.add_argument("--quality", type=int, default=80, help="WebP quality.")
    args = p.parse_args()

    items = _fetch_items(args.base_url)
    print(f"Rendering thumbnails for {len(items)} gallery items...")
    written = page_fallback = skipped = 0
    for it in items:
        paper, page = str(it["paper_id"]), it["page_number"]
        bbox_raw = it.get("bbox")
        bbox = bbox_raw if isinstance(bbox_raw, list) else None
        sid = _safe_id(str(it["chunk_id"]))
        out_dir = args.pages_dir / paper / "thumbs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{sid}.webp"

        crop_path = args.figures_dir / paper / f"{sid}.png"
        try:
            if crop_path.exists():
                thumb = _thumb_from_crop(crop_path, args.max_px)
            elif bbox:
                page_path = args.pages_dir / paper / f"{paper}_p{page}.png"
                if not page_path.exists():
                    skipped += 1
                    continue
                thumb = _thumb_from_page(page_path, bbox, args.max_px)
                page_fallback += 1
            else:
                skipped += 1
                continue
            thumb.save(out, "WEBP", quality=args.quality, method=6)
            written += 1
        except Exception as exc:  # one bad item shouldn't abort the whole run
            print(f"  WARN {it['chunk_id']}: {exc}")
            skipped += 1

    print(f"Wrote {written} thumbnails ({page_fallback} cropped from page), {skipped} skipped.")


if __name__ == "__main__":
    main()
