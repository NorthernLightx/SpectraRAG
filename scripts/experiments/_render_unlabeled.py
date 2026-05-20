"""Render the 4 unlabeled picture-side detections so we can see what
Docling actually picked up."""
from __future__ import annotations
import json, urllib.request, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

PAGES_DIR = Path("data/pages")
OUT = Path("data/unlabeled_inspect")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150
PX = DPI / 72.0
PAD = 20

req = urllib.request.Request(
    "http://localhost:6333/collections/eval_docling_classified_probe/points/scroll",
    data=json.dumps({
        "limit": 500, "with_payload": True, "with_vector": False,
        "filter": {"must": [
            {"key": "metadata.kind", "match": {"value": "figure"}},
            {"key": "metadata.role", "match": {"value": "unlabeled"}},
        ]},
    }).encode(),
    headers={"Content-Type": "application/json"}, method="POST",
)
pts = json.loads(urllib.request.urlopen(req).read())["result"]["points"]
crops = []
for p in pts:
    pl = p["payload"]
    m = pl["metadata"]
    bb = m["bbox"]
    page = pl["page_numbers"][0]
    paper = pl["paper_id"]
    png = PAGES_DIR / paper / f"{paper}_p{page}.png"
    img = Image.open(png).convert("RGB")
    iw, ih = img.size
    x0, y0, x1, y1 = (v * PX for v in bb)
    cx0 = max(0, int(x0 - PAD * PX))
    cy0 = max(0, int(y0 - PAD * PX))
    cx1 = min(iw, int(x1 + PAD * PX))
    cy1 = min(ih, int(y1 + PAD * PX))
    crop = img.crop((cx0, cy0, cx1, cy1)).copy()
    d = ImageDraw.Draw(crop)
    box = (int(x0 - cx0), int(y0 - cy0), int(x1 - cx0), int(y1 - cy0))
    d.rectangle(box, outline=(255, 30, 30), width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    label = f"{pl['chunk_id']}  docling={m.get('docling_label')}({m.get('docling_label_confidence',0):.2f})"
    label_h = 26
    canvas = Image.new("RGB", (max(crop.width, 500), crop.height + label_h), (20, 20, 24))
    canvas.paste(crop, (0, label_h))
    ImageDraw.Draw(canvas).text((6, 4), label, fill=(220, 220, 220), font=font)
    out = OUT / (pl["chunk_id"].replace("::", "__") + ".png")
    canvas.save(out)
    crops.append(canvas)
    print(f"  saved {out}")

# contact sheet
if crops:
    cols = 2
    rows = (len(crops) + cols - 1) // cols
    tw = 720
    scaled = [c.resize((tw, int(c.height * tw / c.width))) for c in crops]
    rh = max(c.height for c in scaled)
    sheet = Image.new("RGB", (cols * tw, rows * rh), (10, 10, 12))
    for i, c in enumerate(scaled):
        r, k = divmod(i, cols)
        sheet.paste(c, (k * tw, r * rh))
    sheet_path = OUT / "_unlabeled_sheet.png"
    sheet.save(sheet_path)
    print(f"\ncontact sheet: {sheet_path}")
