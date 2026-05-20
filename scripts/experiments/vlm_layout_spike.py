"""ADR 0020 kill-spike: can a VLM-as-parser recover ingestion misses?

Same discipline as ADR 0018's GraphRAG spike: don't pivot the ingestion
stack on a hunch. Take the audit-flagged miss pages from 2604.22753v1
(Figures 2/3, Tables 1/3/4 — every miss the overlay tool surfaced) plus
one known-good control page (p02, Figure 1 correctly extracted), render
each at 150 DPI, and ask `qwen3-vl:235b-cloud` (already pulled via
Ollama, matches the cloud-via-Ollama preference) to list every figure
and table as JSON with bboxes + captions.

Continue / kill criteria:
- recovers ≥4 of 5 known misses with plausible bboxes → continue, wire as
  cascade fallback behind extract_figures / extract_tables, ADR 0020;
- recovers ≤2 / 5 or hallucinates on the control page → kill, pivot to
  a deterministic layout-parser library (Docling / Marker) instead.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from pathlib import Path

import fitz
import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT = (
    "You read a single page from a scientific paper rendered at 150 DPI as an "
    "image. List EVERY figure and table on this page as a JSON array. For each "
    'item, return: {"type": "figure" or "table", "label": caption label as '
    'written (e.g. "1", "E.1", "I", empty string if no label), "bbox": [x0, y0, '
    "x1, y1] in image pixel coordinates with origin at top-left, "
    '"caption": full caption text (empty string if none), '
    '"summary": one short sentence on what it shows}.\n'
    "If there are no figures or tables on this page, return [].\n"
    "Output ONLY the JSON array, no prose, no code fences, no commentary."
)

# 2604.22753v1: audit ground truth from the overlay tool.
TARGETS: list[tuple[int, str]] = [
    (2, "CONTROL — Figure 1 should be reported, nothing extra"),
    (6, "MISS — should recover Table 1 (task statistics)"),
    (7, "MISS — should recover Figure 2 (4-panel line plot) and Table 2"),
    (8, "MISS — should recover Figure 3 (t-SNE/parameter-space viz)"),
    (9, "MISS — should recover Table 3 (ablation)"),
    (13, "MISS — should recover Table 4"),
]

OLLAMA = "http://localhost:11434"
MODEL = "qwen3-vl:235b-cloud"
PDF = Path("data/papers/2604.22753v1.pdf")
OUT_DIR = Path("data/eval/ingestion/vlm-spike")
DPI = 150


async def call_vlm(client: httpx.AsyncClient, image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = await client.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2000},
        },
        timeout=600.0,
    )
    if response.status_code != 200:
        return f"<HTTP {response.status_code}: {response.text[:200]}>"
    data = response.json()
    if not isinstance(data, dict):
        return "<non-dict response>"
    return str((data.get("message") or {}).get("content", ""))


def parse_json_array(text: str) -> list[dict] | None:
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j <= i:
        return None
    try:
        data = json.loads(s[i : j + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    print(f"VLM spike: {MODEL} via Ollama  ({len(TARGETS)} pages from {PDF.name})\n")

    async with httpx.AsyncClient() as client:
        with fitz.open(PDF) as doc:
            for page_no, note in TARGETS:
                page = doc[page_no - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72.0, DPI / 72.0), alpha=False)
                img_path = OUT_DIR / f"p{page_no:02d}.png"
                pix.save(str(img_path))
                img_w, img_h = pix.width, pix.height

                print(f"=== p{page_no:02d}  ({img_w}x{img_h})  {note}")
                text = await call_vlm(client, img_path)
                parsed = parse_json_array(text)
                if parsed is None:
                    print(f"  PARSE FAIL.  raw[:240]: {text[:240]!r}\n")
                    results.append(
                        {"page": page_no, "note": note, "items": None, "raw": text[:600]}
                    )
                    continue
                for item in parsed:
                    bb = item.get("bbox")
                    bb_str = (
                        "[{:>4d},{:>4d},{:>4d},{:>4d}]".format(*[int(x) for x in bb])
                        if isinstance(bb, list) and len(bb) == 4
                        else str(bb)[:30]
                    )
                    cap = (item.get("caption") or "")[:64]
                    print(
                        f"  {str(item.get('type', '?')):6}  label={str(item.get('label', '')):>5}"
                        f"  bbox={bb_str}  caption: {cap}"
                    )
                if not parsed:
                    print("  (empty list — VLM saw no figures/tables on this page)")
                print()
                results.append(
                    {
                        "page": page_no,
                        "note": note,
                        "image_size_px": [img_w, img_h],
                        "items": parsed,
                    }
                )

    (OUT_DIR / "spike-results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {OUT_DIR}/spike-results.json")


if __name__ == "__main__":
    asyncio.run(main())
