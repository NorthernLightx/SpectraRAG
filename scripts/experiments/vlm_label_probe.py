"""Compare Docling's DocumentFigureClassifier-v2.5 against a large VLM
(``qwen3-vl:235b-cloud`` via Ollama) on a hand-picked mix from
``eval_docling_classified_probe``:

- the 4 ``unlabeled`` cases (Docling called them table-ish at low conf)
- a handful Docling nailed (logo 1.00, bar_chart 1.00, flow_chart 0.97,
  icon 0.99)
- the rescued small Figure 3 (Docling called it logo, caption said figure)

Each picture's saved crop is sent to the VLM with a short closed-set
prompt. Output: per-item table of Docling vs VLM call + latency, and
a verdict on whether the VLM is worth wiring as a residual layer.

Run:
  python scripts/experiments/vlm_label_probe.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

OLLAMA = "http://localhost:11434"
MODEL = "qwen3-vl:235b-cloud"
COLLECTION = "eval_docling_classified_probe"
QDRANT = "http://localhost:6333"
FIGURES_DIR = Path("data/figures")

# Closed label set the VLM picks from — same vocabulary as Docling so the
# comparison is apples-to-apples. ``prose_callout`` and ``ascii_diagram``
# are added because the probe paper has those (Filesystem Policy, Directory
# Tree) and Docling doesn't have them; we want to see if the VLM volunteers
# them as ``other`` or finds a closer label in its own.
LABELS = [
    "logo", "icon", "signature", "stamp", "bar_code", "qr_code",
    "bar_chart", "box_plot", "flow_chart", "line_chart", "pie_chart",
    "scatter_plot", "photograph", "screenshot_from_computer",
    "screenshot_from_manual", "chemistry_structure", "engineering_drawing",
    "geographical_map", "topographical_map", "table",
    "page_thumbnail", "full_page_image", "calendar", "music",
    "crossword_puzzle", "other",
]

PROMPT = (
    "Classify the image into exactly one of these labels:\n"
    + ", ".join(LABELS)
    + ".\n\n"
    "Reply with a single line of JSON only:\n"
    '{"label": "<one of the labels above>", "confidence": 0.0-1.0, '
    '"why": "<one short sentence>"}\n'
    "No prose outside the JSON. No code fences."
)


def fetch_figure_chunks() -> list[dict]:
    payload = {
        "limit": 500, "with_payload": True, "with_vector": False,
        "filter": {"must": [{"key": "metadata.kind", "match": {"value": "figure"}}]},
    }
    req = urllib.request.Request(
        f"{QDRANT}/collections/{COLLECTION}/points/scroll",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["result"]["points"]


def pick_samples(pts: list[dict]) -> list[dict]:
    """Pick a mix: all unlabeled + one of each Docling label observed."""
    keep: list[dict] = []
    seen_labels: set[str] = set()
    # unlabeled first
    for p in pts:
        m = p["payload"].get("metadata", {})
        if m.get("role") == "unlabeled":
            keep.append(p)
    # then one example per Docling label not yet sampled
    for p in pts:
        m = p["payload"].get("metadata", {})
        lbl = m.get("docling_label")
        if lbl is None or lbl in seen_labels:
            continue
        if p in keep:
            continue
        keep.append(p)
        seen_labels.add(lbl)
    # the rescued small Figure 3 — small picture with paper-figure caption
    for p in pts:
        m = p["payload"].get("metadata", {})
        bb = m.get("bbox") or [0, 0, 0, 0]
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        cap = p["payload"].get("text", "") or ""
        if area < 2000 and cap.lower().startswith("figure 3"):
            if p not in keep:
                keep.append(p)
            break
    return keep


def call_vlm(image_path: Path) -> tuple[str, dict, float]:
    if not image_path.exists():
        return "<missing>", {}, 0.0
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return f"<HTTP {e.code}>", {}, time.perf_counter() - t0
    except Exception as e:
        return f"<err {type(e).__name__}: {e}>", {}, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    content = (data.get("message") or {}).get("content", "") or ""
    return content, data, elapsed


def parse_json(content: str) -> dict:
    # The VLM occasionally wraps in ``` fences despite instructions; strip them.
    s = content.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(s)
    except Exception:
        return {"label": "<parse-fail>", "confidence": 0.0, "why": s[:80]}


def main() -> None:
    pts = fetch_figure_chunks()
    samples = pick_samples(pts)
    print(f"sampled {len(samples)} chunks from {len(pts)} total")
    print()
    total_time = 0.0
    total_tokens = 0
    print(f"{'chunk_id':40s}  {'docling':30s}  {'vlm':30s}  {'why':50s}  {'latency':>8s}")
    print("-" * 165)
    for p in samples:
        pl = p["payload"]
        cid = pl["chunk_id"]
        m = pl.get("metadata", {})
        docling_lbl = m.get("docling_label")
        docling_conf = m.get("docling_label_confidence", 0.0)
        docling_str = f"{docling_lbl or '-'}({docling_conf:.2f})"
        # image path: data/figures/<paper>/<chunk_id with :→_>.png
        paper_id = pl["paper_id"]
        img = FIGURES_DIR / paper_id / (cid.replace("::", "_") + ".png")
        if not img.exists():
            # fallback: try alt naming
            img = FIGURES_DIR / paper_id / (cid.replace("::", "__") + ".png")
        content, data, elapsed = call_vlm(img)
        total_time += elapsed
        parsed = parse_json(content)
        vlm_lbl = parsed.get("label", "?")
        vlm_conf = float(parsed.get("confidence", 0.0) or 0.0)
        vlm_why = (parsed.get("why") or "")[:50]
        token_count = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        total_tokens += token_count
        print(f"{cid:40s}  {docling_str:30s}  {vlm_lbl}({vlm_conf:.2f}):30  {vlm_why:50s}  {elapsed:6.1f}s")
    print()
    print(f"total: {len(samples)} calls, {total_time:.1f}s wall time, {total_tokens} tokens")


if __name__ == "__main__":
    main()
