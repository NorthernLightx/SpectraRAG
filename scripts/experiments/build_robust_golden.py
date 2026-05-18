"""Build `data/golden/robust-v1.yaml` — a routing-fair evaluation set.

Why: v3 (arXiv) is text-heavy and saturated; MMLongBench is ~93 % visual so
it rewards a degenerate "always-visual" policy (ADR 0013). Neither can judge
a *router* honestly. A routing-fair set must be **balanced by true evidence
modality** so no lazy policy can win: "always-text" fails the visual half,
"always-visual" fails the text half, only correct per-query routing scores
high. Combined with the measured per-leg asymmetry (text leg ≈0 on visual
evidence; visual leg loses ~10 % on genuine text, ADR 0007), modality
balance makes the set routing-fair *by construction*.

Method: stratified-sample the existing **human labels** (no LLM-invented
golden). MMLongBench's `note` carries `evidence_sources` (the true evidence
location); v3 supplies production-domain (arXiv) text questions. Scoring is
unified to page level; v3 chunk-ids → pages via the `::pN` convention (same
as scripts/rescore_mmlb_pages). The eval that consumes this must run
**paper-filtered** retrieval (every query carries its paper_id) so the two
corpora's page numbers can't collide — a deliberate, recorded choice.

Output: data/golden/robust-v1.yaml + a printed composition self-check.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MMLB = ROOT / "data" / "golden" / "mmlongbench-v1.yaml"
V3 = ROOT / "data" / "golden" / "v3.yaml"
OUT = ROOT / "data" / "golden" / "robust-v1.yaml"

VISUAL = {"Chart", "Figure", "Table"}
TEXTUAL = {"Pure-text (Plain-text)", "Generalized-text (Layout)"}
_EV = re.compile(r"evidence_sources=(\[[^\]]*\])")
_TOKEN = re.compile(r"'([^']*)'")
_PAGE = re.compile(r"::p(\d+)")
# per-bucket target; capped per source so each bucket is a genuine mix
TARGET = {"text": 28, "figure": 26, "table": 20, "mixed": 16, "ooc": 14}


def _evidence(note: str | None) -> list[str]:
    m = _EV.search(note or "")
    return _TOKEN.findall(m.group(1)) if m else []


def _bucket_mmlb(q: dict) -> str:
    ev = set(_evidence(q.get("note")))
    if not ev:
        return "ooc"
    has_v, has_t = ev & VISUAL, ev & TEXTUAL
    if has_v and has_t:
        return "mixed"
    if has_v:
        return "table" if has_v == {"Table"} else "figure"
    return "text"


def _bucket_v3(q: dict) -> str:
    c = q.get("category")
    if c == "out_of_corpus":
        return "ooc"
    if c in ("figure", "table"):
        return c
    return "text"  # factual/definitional/equation/multi_hop on an arXiv text corpus


def _pages_from_chunks(chunk_ids: list[str]) -> list[int]:
    out: list[int] = []
    for cid in chunk_ids:
        m = _PAGE.search(cid)
        if m and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def main() -> None:
    rng = random.Random(20260518)  # reproducible selection
    mmlb = yaml.safe_load(MMLB.read_text(encoding="utf-8"))["queries"]
    v3 = yaml.safe_load(V3.read_text(encoding="utf-8"))["queries"]

    pools: dict[str, list[dict]] = {b: [] for b in TARGET}
    for q in mmlb:
        b = _bucket_mmlb(q)
        pools[b].append(
            {
                "query_id": f"mmlb__{q['query_id']}",
                "text": q["text"],
                "paper_id": q["paper_id"],
                "category": q["category"],
                "relevant_chunk_ids": [],
                "relevant_pages": list(q.get("relevant_pages") or []),
                "expected_facts": list(q.get("expected_facts") or []),
                "note": f"src=mmlongbench bucket={b} ev={_evidence(q.get('note'))}",
            }
        )
    for q in v3:
        b = _bucket_v3(q)
        pools[b].append(
            {
                "query_id": f"v3__{q['query_id']}",
                "text": q["text"],
                "paper_id": q["paper_id"],
                "category": q["category"],
                "relevant_chunk_ids": list(q.get("relevant_chunk_ids") or []),
                "relevant_pages": _pages_from_chunks(q.get("relevant_chunk_ids") or []),
                "expected_facts": list(q.get("expected_facts") or []),
                "note": f"src=v3 bucket={b} ev=arxiv-{q['category']}",
            }
        )

    chosen: list[dict] = []
    comp: dict[str, dict[str, int]] = {}
    for b, n in TARGET.items():
        pool = pools[b]
        rng.shuffle(pool)
        mmlb_q = [x for x in pool if x["query_id"].startswith("mmlb__")]
        v3_q = [x for x in pool if x["query_id"].startswith("v3__")]
        take: list[dict] = []
        while len(take) < n and (mmlb_q or v3_q):
            if v3_q and (len(take) % 2 == 0 or not mmlb_q):
                take.append(v3_q.pop())
            elif mmlb_q:
                take.append(mmlb_q.pop())
        chosen.extend(take)
        comp[b] = {
            "total": len(take),
            "v3": sum(1 for x in take if x["query_id"].startswith("v3__")),
            "mmlb": sum(1 for x in take if x["query_id"].startswith("mmlb__")),
            "pool": len(pool),
        }

    doc = {"name": "robust", "version": "v1", "queries": chosen}
    OUT.write_text(
        "# Routing-fair eval: modality-balanced, built by scripts/experiments/build_robust_golden.py\n"
        "# (ADR 0015). Score paper-filtered + page-level (queries carry paper_id;\n"
        "# cross-corpus page numbers must not collide).\n"
        + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )

    print(f"Wrote {OUT}  ({len(chosen)} queries)")
    print(f"{'bucket':8s}{'total':>7}{'v3':>5}{'mmlb':>6}{'pool':>7}")
    for b in TARGET:
        c = comp[b]
        print(f"{b:8s}{c['total']:>7}{c['v3']:>5}{c['mmlb']:>6}{c['pool']:>7}")
    vis = comp["figure"]["total"] + comp["table"]["total"]
    print(
        f"\nrouting-fairness: text={comp['text']['total']} vs "
        f"visual={vis} vs mixed={comp['mixed']['total']} vs ooc={comp['ooc']['total']}"
        " — balanced => no always-X policy can win."
    )


if __name__ == "__main__":
    main()
