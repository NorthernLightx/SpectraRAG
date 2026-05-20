"""One-off: read the classified figures from the test collection."""
from __future__ import annotations
import json, urllib.request, sys
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

req = urllib.request.Request(
    "http://localhost:6333/collections/eval_docling_classified_probe/points/scroll",
    data=json.dumps({
        "limit": 500, "with_payload": True, "with_vector": False,
        "filter": {"must": [{"key": "metadata.kind", "match": {"value": "figure"}}]},
    }).encode(),
    headers={"Content-Type": "application/json"}, method="POST",
)
pts = json.loads(urllib.request.urlopen(req).read())["result"]["points"]
print(f"figure chunks: {len(pts)}")

role_c, label_c = Counter(), Counter()
for p in pts:
    m = p["payload"].get("metadata", {})
    role_c[m.get("role", "-")] += 1
    label_c[m.get("docling_label", "-")] += 1

print("\nroles:")
for k, v in role_c.most_common():
    print(f"  {k}: {v}")

print("\ndocling labels (top 12):")
for k, v in label_c.most_common(12):
    print(f"  {k}: {v}")

print("\nsamples by docling label:")
seen = set()
for p in pts:
    m = p["payload"].get("metadata", {})
    lbl = m.get("docling_label", "-")
    if lbl in seen:
        continue
    seen.add(lbl)
    conf = m.get("docling_label_confidence", 0)
    cid = p["payload"]["chunk_id"]
    role = m.get("role")
    print(f"  {cid}  label={lbl}({conf:.2f})  role={role}")
