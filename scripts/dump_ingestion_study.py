"""Dump a Qdrant collection's ingestion artefacts to disk for manual study.

Reads every chunk from `--collection`, writes:

    data/ingestion_study/<collection>/
        README.md                            human-friendly index
        summary.json                         aggregate counts
        per_paper/<paper_id>/
            figures.jsonl                    one figure chunk per line
            tables.jsonl                     one table chunk per line
            pages_with_overlays/
                p<N>.png                     source page with role-coloured bboxes
        by_label/<docling_label>/
            contact_sheet.png                grid of crops, captions underneath
        unlabeled/<paper>__<chunk>.png       individual bbox crops, focus bucket

Color-coding:
    figure       -> blue   (accent — real publication content)
    table        -> green  (real publication content, separate kind)
    decoration   -> grey   (logos / icons / signatures — gallery hides)
    unlabeled    -> orange (worth a human eye)

Local-only artefact — `data/ingestion_study/` is gitignored.

Run:
    python scripts/dump_ingestion_study.py --collection eval_docling_classified
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

QDRANT_DEFAULT = "http://localhost:6333"
PAGES_DIR = Path("data/pages")
DPI = 150
PX_PER_PT = DPI / 72.0
ROLE_COLOR: dict[str, tuple[int, int, int]] = {
    "figure": (88, 166, 255),  # accent blue
    "table": (63, 185, 80),  # ok green
    "decoration": (139, 148, 158),  # muted grey
    "unlabeled": (255, 159, 47),  # warning orange
}


def fetch_all_chunks(qdrant_url: str, collection: str) -> list[dict[str, Any]]:
    pts: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        payload: dict[str, Any] = {
            "limit": 256,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            payload["offset"] = offset
        req = urllib.request.Request(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        pts.extend(body["result"]["points"])
        offset = body["result"].get("next_page_offset")
        if offset is None:
            break
    return pts


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_page_overlay(
    page_png: Path,
    boxes: list[tuple[list[float], str, str]],  # (bbox_pts, role, label)
    out_path: Path,
) -> None:
    """Draw every figure/table bbox on a copy of the page, color by role."""
    img = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(14)
    for bbox, role, label in boxes:
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (v * PX_PER_PT for v in bbox)
        color = ROLE_COLOR.get(role, (200, 200, 200))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        # Label tag above the box
        tag = f"{role}/{label}" if label else role
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        tx, ty = x0, max(0, y0 - th - 4)
        draw.rectangle((tx, ty, tx + tw + 6, ty + th + 4), fill=color)
        draw.text((tx + 3, ty + 2), tag, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def crop_around_bbox(page_png: Path, bbox_pts: list[float], pad_pt: float = 20.0) -> Image.Image:
    img = Image.open(page_png).convert("RGB")
    iw, ih = img.size
    x0, y0, x1, y1 = (v * PX_PER_PT for v in bbox_pts)
    cx0 = max(0, int(x0 - pad_pt * PX_PER_PT))
    cy0 = max(0, int(y0 - pad_pt * PX_PER_PT))
    cx1 = min(iw, int(x1 + pad_pt * PX_PER_PT))
    cy1 = min(ih, int(y1 + pad_pt * PX_PER_PT))
    crop = img.crop((cx0, cy0, cx1, cy1)).copy()
    d = ImageDraw.Draw(crop)
    box = (int(x0 - cx0), int(y0 - cy0), int(x1 - cx0), int(y1 - cy0))
    d.rectangle(box, outline=(255, 30, 30), width=2)
    return crop


def label_card(crop: Image.Image, caption: str, header: str) -> Image.Image:
    """Top a crop with a header bar and bottom-paste a caption block."""
    header_h, footer_h = 22, 60
    width = max(crop.width, 360)
    canvas = Image.new("RGB", (width, header_h + crop.height + footer_h), (20, 20, 24))
    # Header
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), header, fill=(220, 220, 220), font=_font(12))
    # Crop centred horizontally
    canvas.paste(crop, ((width - crop.width) // 2, header_h))
    # Footer: caption, wrapped to fit
    cap = caption.replace("\n", " ").strip()
    if len(cap) > 220:
        cap = cap[:217] + "..."
    # Naive wrap
    lines = []
    line = ""
    for word in cap.split():
        trial = (line + " " + word).strip()
        if d.textbbox((0, 0), trial, font=_font(11))[2] > width - 12:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    for i, ln in enumerate(lines[:4]):
        d.text((6, header_h + crop.height + 4 + i * 13), ln, fill=(200, 200, 200), font=_font(11))
    return canvas


def build_contact_sheet(cards: list[Image.Image], cols: int = 4, gap: int = 6) -> Image.Image:
    if not cards:
        return Image.new("RGB", (200, 60), (20, 20, 24))
    cw = max(c.width for c in cards)
    rh = max(c.height for c in cards)
    rows = (len(cards) + cols - 1) // cols
    width = cols * cw + (cols + 1) * gap
    height = rows * rh + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (10, 10, 12))
    for i, c in enumerate(cards):
        r, k = divmod(i, cols)
        x = gap + k * (cw + gap)
        y = gap + r * (rh + gap)
        sheet.paste(c, (x, y))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant", default=QDRANT_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=Path("data/ingestion_study"))
    ap.add_argument(
        "--max-per-label",
        type=int,
        default=12,
        help="Cap thumbnails per label in the by_label contact sheets.",
    )
    args = ap.parse_args()

    out_dir = args.out_root / args.collection
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dumping to {out_dir}")

    pts = fetch_all_chunks(args.qdrant, args.collection)
    print(f"total chunks in collection: {len(pts)}")

    # Bucketing
    by_paper_figures: dict[str, list[dict]] = defaultdict(list)
    by_paper_tables: dict[str, list[dict]] = defaultdict(list)
    by_label: dict[str, list[dict]] = defaultdict(list)
    role_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    vlm_caption_count = 0

    for p in pts:
        pl = p["payload"]
        m = pl.get("metadata", {})
        kind = m.get("kind")
        if kind not in {"figure", "table"}:
            continue  # text chunks aren't part of this study
        kind_counts[kind] += 1
        rec = {
            "chunk_id": pl["chunk_id"],
            "paper_id": pl["paper_id"],
            "page_numbers": pl.get("page_numbers", []),
            "text": pl.get("text", ""),
            "kind": kind,
            "role": m.get("role"),
            "docling_label": m.get("docling_label"),
            "docling_label_confidence": m.get("docling_label_confidence"),
            "has_vlm_caption": bool(m.get("has_vlm_caption")),
            "bbox": m.get("bbox"),
        }
        if rec["has_vlm_caption"]:
            vlm_caption_count += 1
        role_counts[rec["role"] or "?"] += 1
        if rec["docling_label"]:
            label_counts[rec["docling_label"]] += 1
        by_label[rec["docling_label"] or f"_role_{rec['role'] or 'unknown'}"].append(rec)
        if kind == "table":
            by_paper_tables[rec["paper_id"]].append(rec)
        else:
            by_paper_figures[rec["paper_id"]].append(rec)

    # ---- per-paper JSONL + page overlays ----
    per_paper_dir = out_dir / "per_paper"
    per_paper_dir.mkdir(exist_ok=True)
    paper_ids = sorted(set(by_paper_figures) | set(by_paper_tables))
    print(f"papers: {len(paper_ids)}")
    for pid in paper_ids:
        pdir = per_paper_dir / pid
        pdir.mkdir(exist_ok=True)
        with (pdir / "figures.jsonl").open("w", encoding="utf-8") as fh:
            for r in sorted(by_paper_figures.get(pid, []), key=lambda r: r["chunk_id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (pdir / "tables.jsonl").open("w", encoding="utf-8") as fh:
            for r in sorted(by_paper_tables.get(pid, []), key=lambda r: r["chunk_id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Per-page overlays: group boxes by page
        boxes_by_page: dict[int, list[tuple[list[float], str, str]]] = defaultdict(list)
        for r in by_paper_figures.get(pid, []) + by_paper_tables.get(pid, []):
            if not r["bbox"] or not r["page_numbers"]:
                continue
            page = r["page_numbers"][0]
            label = r["docling_label"] or ""
            boxes_by_page[page].append((r["bbox"], r["role"] or "unlabeled", label))
        overlay_dir = pdir / "pages_with_overlays"
        overlay_dir.mkdir(exist_ok=True)
        for page_no, boxes in boxes_by_page.items():
            src_png = PAGES_DIR / pid / f"{pid}_p{page_no}.png"
            if not src_png.exists():
                continue
            render_page_overlay(src_png, boxes, overlay_dir / f"p{page_no:03d}.png")

    # ---- by-label contact sheets ----
    by_label_dir = out_dir / "by_label"
    by_label_dir.mkdir(exist_ok=True)
    for label, items in sorted(by_label.items()):
        cards: list[Image.Image] = []
        for r in items[: args.max_per_label]:
            if not r["bbox"] or not r["page_numbers"]:
                continue
            src_png = PAGES_DIR / r["paper_id"] / f"{r['paper_id']}_p{r['page_numbers'][0]}.png"
            if not src_png.exists():
                continue
            try:
                crop = crop_around_bbox(src_png, r["bbox"])
            except Exception:
                continue
            crop.thumbnail((280, 280))
            header = f"{r['paper_id']}::p{r['page_numbers'][0]}  role={r['role']}"
            cap = (r["text"] or "").strip()
            if cap.startswith("["):
                cap = "(no caption)"
            cards.append(label_card(crop, cap, header))
        sheet = build_contact_sheet(cards, cols=3)
        sheet_dir = by_label_dir / label
        sheet_dir.mkdir(exist_ok=True)
        sheet.save(sheet_dir / "contact_sheet.png")

    # ---- focused: every unlabeled / VLM-captioned chunk gets its own card ----
    focus_dir = out_dir / "unlabeled_focus"
    focus_dir.mkdir(exist_ok=True)
    focus_items = [
        r
        for items in by_paper_figures.values()
        for r in items
        if r["role"] == "unlabeled" or r["has_vlm_caption"]
    ]
    focus_items.sort(key=lambda r: r["chunk_id"])
    for r in focus_items:
        if not r["bbox"] or not r["page_numbers"]:
            continue
        src_png = PAGES_DIR / r["paper_id"] / f"{r['paper_id']}_p{r['page_numbers'][0]}.png"
        if not src_png.exists():
            continue
        crop = crop_around_bbox(src_png, r["bbox"], pad_pt=30)
        header = (
            f"{r['chunk_id']}  role={r['role']}  "
            f"docling={r['docling_label']}({(r['docling_label_confidence'] or 0):.2f})"
        )
        card = label_card(crop, r["text"], header)
        card.save(focus_dir / (r["chunk_id"].replace("::", "__") + ".png"))

    # ---- summary.json + README.md ----
    summary = {
        "collection": args.collection,
        "total_chunks_in_collection": len(pts),
        "papers": len(paper_ids),
        "kinds": dict(kind_counts),
        "roles": dict(role_counts),
        "docling_labels": dict(label_counts),
        "vlm_captioned": vlm_caption_count,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# Ingestion study — `{args.collection}`

Generated by `scripts/dump_ingestion_study.py`. Local-only, not committed.

## Counts

- papers: {summary["papers"]}
- total chunks in collection: {summary["total_chunks_in_collection"]}
- figure-kind chunks: {kind_counts.get("figure", 0)}
- table-kind chunks: {kind_counts.get("table", 0)}
- VLM-captioned figures: {vlm_caption_count}

### Roles
{chr(10).join(f"- **{r}**: {c}" for r, c in sorted(role_counts.items()))}

### Docling labels (figure subtypes)
{chr(10).join(f"- **{label}**: {c}" for label, c in sorted(label_counts.items(), key=lambda x: -x[1]))}

## Folders

- `per_paper/<paper_id>/figures.jsonl` — one JSON object per figure chunk
  (role, docling_label, confidence, caption, vlm_caption, bbox, …).
- `per_paper/<paper_id>/tables.jsonl` — same shape for tables.
- `per_paper/<paper_id>/pages_with_overlays/p<N>.png` — page rendered at
  150 DPI with every figure/table bbox drawn, colour-coded by role
  (blue=figure, green=table, grey=decoration, orange=unlabeled).
- `by_label/<label>/contact_sheet.png` — grid of up to {args.max_per_label}
  cropped thumbnails per Docling label (`logo`, `bar_chart`, `flow_chart`,
  …), with caption text below each.
- `unlabeled_focus/<chunk_id>.png` — every chunk that ended up `role=unlabeled`
  OR carries a VLM-generated caption, each with its full text on the card.
  Useful for eyeballing the residual bucket.
- `summary.json` — machine-readable version of the counts above.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"summary: {summary}")
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
