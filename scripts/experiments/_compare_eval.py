"""Compare two eval run JSONs side-by-side."""
from __future__ import annotations
import json, sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

BASELINE = "data/eval/baseline-docling-text.json"
NEW = "data/eval/runs/run-20260520-232623.json"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(d: dict) -> dict:
    pqs = d["per_query"]
    metrics = ["answer_correctness", "faithfulness", "answer_relevance", "context_precision"]
    out: dict = {"n": len(pqs)}
    for m in metrics:
        vals = [pq.get("generation", {}).get(m) for pq in pqs if pq.get("generation", {}).get(m) is not None]
        out[m] = sum(vals) / len(vals) if vals else None
        out[f"{m}_n"] = len(vals)
    cit = [pq.get("generation", {}).get("citation_count", 0) for pq in pqs if pq.get("generation")]
    out["cited_queries"] = sum(1 for c in cit if c and c > 0)
    return out


def per_category(d: dict, metric: str = "answer_correctness") -> dict:
    """answer_correctness per query category, where category comes from the
    golden record (each per_query carries category)."""
    by_cat: dict[str, list[float]] = {}
    for pq in d["per_query"]:
        cat = pq.get("category") or pq.get("golden", {}).get("category") or "?"
        v = pq.get("generation", {}).get(metric)
        if v is None:
            continue
        by_cat.setdefault(cat, []).append(v)
    return {k: sum(v) / len(v) for k, v in sorted(by_cat.items())}


baseline = load(BASELINE)
new = load(NEW)

print(f"baseline run_id: {baseline['run_id']}")
print(f"new run_id:      {new['run_id']}")
print()

base_a = aggregate(baseline)
new_a = aggregate(new)

print(f"{'metric':<22}{'baseline':>12}{'new':>12}{'delta':>12}{'% rel':>10}")
print("-" * 70)
for m in ["answer_correctness", "faithfulness", "answer_relevance", "context_precision"]:
    b, n = base_a[m], new_a[m]
    if b is None or n is None:
        print(f"{m:<22}{b!r:>12}{n!r:>12}")
        continue
    d = n - b
    pct = 100 * d / b if b else 0
    print(f"{m:<22}{b:>12.4f}{n:>12.4f}{d:>+12.4f}{pct:>+9.1f}%")
print()
print(f"{'cited queries':<22}{base_a['cited_queries']:>12}{new_a['cited_queries']:>12}")
print(f"{'n queries':<22}{base_a['n']:>12}{new_a['n']:>12}")

print()
print("=== answer_correctness per category ===")
b_cats = per_category(baseline)
n_cats = per_category(new)
all_cats = sorted(set(b_cats) | set(n_cats))
print(f"{'category':<22}{'baseline':>12}{'new':>12}{'delta':>12}")
print("-" * 60)
for c in all_cats:
    b = b_cats.get(c)
    n = n_cats.get(c)
    if b is None or n is None:
        bs = f"{b:.4f}" if b is not None else "-"
        ns = f"{n:.4f}" if n is not None else "-"
        print(f"{c:<22}{bs:>12}{ns:>12}")
        continue
    print(f"{c:<22}{b:>12.4f}{n:>12.4f}{n - b:>+12.4f}")
