"""One-off: list the 4 unlabeled figure chunks for inspection."""
from __future__ import annotations
import json, urllib.request, sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

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
print(f"unlabeled: {len(pts)}")
for p in pts:
    pl = p["payload"]
    m = pl.get("metadata", {})
    bb = m.get("bbox") or [0, 0, 0, 0]
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    area = w * h
    lbl = m.get("docling_label")
    conf = m.get("docling_label_confidence", 0.0)
    cap = (pl.get("text", "") or "")[:60]
    print(f"  {pl['chunk_id']:42s}  label={lbl}({conf:.2f})  area={area:>7.0f}  cap={cap!r}")
