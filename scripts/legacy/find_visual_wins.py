"""Find queries where the router (hybrid) clearly beat text-only retrieval.

Reads both run JSONs, normalises retrieved chunk-ids to pages, compares
against the golden's relevant_pages, and prints the queries where:
  text-only recall@10 == 0   (text leg missed entirely)
  router  recall@10 >= 0.5   (visual leg recovered)

These are the cleanest "why multi-modal matters" cases — text retrieval
returned nothing relevant in its top-10, hybrid found it.
"""

import json
import re
import sys

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

with open("data/eval/runs/run-20260505-220646.json", encoding="utf-8") as fh:
    text_run = json.load(fh)
with open("data/eval/runs/run-20260506-024716.json", encoding="utf-8") as fh:
    rtr_run = json.load(fh)
with open("data/golden/mmlongbench-v1.yaml", encoding="utf-8") as fh:
    golden = yaml.safe_load(fh)

gold = {
    q["query_id"]: {
        "pages": set(q.get("relevant_pages") or []),
        "category": q.get("category"),
        "text": q.get("text"),
        "expected": (q.get("expected_facts") or [""])[0],
    }
    for q in golden["queries"]
}

_PAGE_RE = re.compile(r"::p(\d+)")


def pages(retrieved: list[str]) -> list[int]:
    seen = set()
    out = []
    for cid in retrieved:
        m = _PAGE_RE.search(cid)
        if not m:
            continue
        p = int(m.group(1))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def recall(retrieved_pages: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(1 for p in retrieved_pages[:k] if p in relevant) / len(relevant)


text_by_qid = {pq["query_id"]: pq for pq in text_run["per_query"]}
rtr_by_qid = {pq["query_id"]: pq for pq in rtr_run["per_query"]}

clean_wins: list[tuple[str, float, float]] = []
for qid in text_by_qid:
    info = gold.get(qid)
    if not info or not info["pages"]:
        continue
    t = pages(text_by_qid[qid].get("retrieved_chunk_ids", []))
    r = pages(rtr_by_qid.get(qid, {}).get("retrieved_chunk_ids", []))
    t_rec = recall(t, info["pages"])
    r_rec = recall(r, info["pages"])
    if t_rec == 0 and r_rec >= 0.5:
        clean_wins.append((qid, t_rec, r_rec))

print(f"=== Clean visual wins: {len(clean_wins)} queries ===")
print("(text recall@10 == 0, router recall@10 >= 0.5)")
print()
for qid, tr, rr in sorted(clean_wins, key=lambda kv: -kv[2]):
    info = gold[qid]
    t_pages = pages(text_by_qid[qid]["retrieved_chunk_ids"])[:10]
    r_pages = pages(rtr_by_qid[qid]["retrieved_chunk_ids"])[:10]
    print(f"--- {qid} ({info['category']}) ---")
    print(f"  query     : {info['text']}")
    print(f"  expected  : {info['expected'][:80]}")
    print(f"  gold pages: {sorted(info['pages'])}")
    print(f"  text top-10 pages: {t_pages}    (recall@10={tr:.2f})")
    print(f"  rtr  top-10 pages: {r_pages}    (recall@10={rr:.2f})")
    # Did router go hybrid for this one?
    rtr_pq = rtr_by_qid[qid]
    print(f"  text answer: {(text_by_qid[qid].get('answer_text') or '')[:120]!r}")
    print(f"  rtr  answer: {(rtr_pq.get('answer_text') or '')[:120]!r}")
    print()
