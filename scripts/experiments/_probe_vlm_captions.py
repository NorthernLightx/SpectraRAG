"""Read back the VLM-captioned chunks from the probe collection so we
can see what qwen3-vl-cloud actually wrote."""
from __future__ import annotations
import json, urllib.request, sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

req = urllib.request.Request(
    "http://localhost:6333/collections/eval_docling_captioned_probe/points/scroll",
    data=json.dumps({
        "limit": 500, "with_payload": True, "with_vector": False,
        "filter": {"must": [
            {"key": "metadata.kind", "match": {"value": "figure"}},
            {"key": "metadata.has_vlm_caption", "match": {"value": True}},
        ]},
    }).encode(),
    headers={"Content-Type": "application/json"}, method="POST",
)
pts = json.loads(urllib.request.urlopen(req).read())["result"]["points"]
print(f"figures with vlm_caption: {len(pts)}")
print()
for p in pts:
    pl = p["payload"]
    m = pl.get("metadata", {})
    print(f"--- {pl['chunk_id']}")
    print(f"  docling_label: {m.get('docling_label')}({m.get('docling_label_confidence', 0):.2f})")
    print(f"  role:          {m.get('role')}")
    print(f"  text (=VLM):   {pl.get('text', '')[:300]}")
    print()
