"""Direct extraction-quality metric: does the offline structured extraction actually
CAPTURE the gold answer? (Goal 2026-06-01: SOTA-level structured-object extraction.)

The QA-lift probe (struct_extract_probe.py) measures whether feeding extracted text
HELPS the reader. This measures the upstream thing: extraction RECALL — for each
card whose answer is a structured-object value (table cell, chart data point), is
that value present in the extracted structured text the model produced offline?

This is a proxy for table/chart-to-text SOTA (e.g. OmniDocBench TEDS, chart RMS-F1)
that needs no external HTML ground truth — it uses the MMLongBench gold value as the
target token and asks "did our extraction surface it". High recall = the structured
object was extracted well; low recall = the extractor missed it (iterate the extractor).

NOT authoring gold: the target is the human MMLongBench gold value; this only checks
presence in model-produced extraction text.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.extraction_recall \
        --cache data/eval/runs/struct_extract_cache.json \
        --failures docs/research/2026-05-29-agenda/postret_failures.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _norm_num(s: str) -> str:
    """Normalise a numeric token: strip %, $, spaces, trailing units, commas."""
    s = s.lower().strip().rstrip("%").strip()
    s = re.sub(r"[,$]", "", s)
    s = re.sub(r"\s*(miles|mile|million|m|k)\b", "", s)
    return s.strip()


def _gold_tokens(gold: str) -> list[str]:
    """Split a gold answer into the value tokens we look for in the extraction.

    List golds arrive as Python/JSON-ish strings with mixed quotes and embedded
    apostrophes (e.g. ["Singapore-Cambridge GCE 'A' Level", ...]) that defeat a
    naive json.loads. Try ast.literal_eval first, then a quoted-substring fallback,
    so each list element becomes its own token rather than one unmatchable blob."""
    import ast

    g = gold.strip()
    if g.startswith("["):
        for parse in (ast.literal_eval, lambda s: json.loads(s.replace("'", '"'))):
            try:
                items = parse(g)
                if isinstance(items, list):
                    return [str(x).strip() for x in items if str(x).strip()]
            except Exception:
                continue
        # last resort: pull quoted substrings
        quoted = re.findall(r"""['"]([^'"]{2,})['"]""", g)
        if quoted:
            return [q.strip() for q in quoted]
    return [g]


def _present(token: str, text: str) -> bool:
    """Is `token` present in `text` (numeric- and punctuation-robust)? For numbers,
    compare normalised forms so '0-375 miles' matches 'red = 0 - 375'. For strings,
    punctuation-insensitive substring (apostrophes/quotes/commas stripped from both)."""
    tok = token.strip()
    if not tok:
        return False
    # purely numeric/range token: each number must appear in the extraction's numbers
    if re.sub(r"[\d\s.\-%,]", "", tok) == "":
        nums = re.findall(r"-?\d+\.?\d*", tok)
        text_nums = {_norm_num(n) for n in re.findall(r"-?\d+\.?\d*", text)}
        return bool(nums) and all(_norm_num(n) in text_nums for n in nums)
    # string token: collapse to alnum+space on both sides (kills apostrophes, quotes, punctuation),
    # then require a WORD-BOUNDARY match so "Yes" does not match "Yesterday" / "SGD" not "USGD".
    def squash(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
    clean = squash(tok)
    if not clean:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(clean)}(?![a-z0-9])", squash(text)) is not None


def gold_reliability(gold: str) -> str:
    """How trustworthy is a presence-match on this gold value as an extraction signal?
    'low' = a single digit/char (matches incidentally inside other tokens — the matcher
    is unreliable, exclude from headline recall); 'ok' otherwise. Surfaced so recall
    numbers can be reported on the reliable subset, per the 2026-06-02 review."""
    stripped = re.sub(r"[^A-Za-z0-9]", "", gold)
    return "low" if len(stripped) <= 1 else "ok"


# Golds that are DERIVED (a count, a computed gap/average) rather than a value printed
# verbatim on the page. These belong to the QA/reasoning stage, not extraction recall —
# the extractor surfacing the raw cells is success even if the derived answer isn't a token.
def _is_derived(f: dict[str, Any]) -> bool:
    q = f.get("query", "").lower()
    derived_cues = ["how many", "average", "gap between", "difference", "more ", "percentage of",
                    "total number", "sum of", "how much more"]
    return any(c in q for c in derived_cues)


# A gold is a STRUCTURED-OBJECT target only if its answer lives in a table/chart value.
# Text-passage answers (definitions, prose) and photo/figure-semantic answers (which
# photo has no person, what colour) are NOT transcription targets — extraction recall
# over them measures the wrong thing. Restrict to table/chart numeric+label golds.
def _is_structured_target(f: dict[str, Any]) -> bool:
    if _is_derived(f):
        return False
    cat = f.get("category", "")
    gold = str(f.get("gold", "")).strip()
    if cat == "table":
        return True
    if cat == "figure":
        # chart-data figures have numeric/label golds; colour/photo-semantic golds don't.
        q = f.get("query", "").lower()
        is_colour = "colour" in q or "color" in q
        has_num = bool(re.search(r"\d", gold))
        return has_num and not is_colour
    return False  # factual/text-passage/out_of_corpus: not a transcription target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/struct_extract_cache.json"))
    ap.add_argument("--failures", type=Path, default=Path("docs/research/2026-05-29-agenda/postret_failures.json"))
    ap.add_argument("--extract-model", default="qwen3-vl:235b-cloud")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    fails = {f["qid"]: f for f in json.loads(args.failures.read_text(encoding="utf-8"))}

    rows: list[dict[str, Any]] = []
    for qid, f in fails.items():
        ek = f"extract::{args.extract_model}::{qid}"
        extract = cache.get(ek)
        if extract is None:
            continue  # not yet extracted
        toks = _gold_tokens(str(f["gold"]))
        hit = all(_present(t, extract) for t in toks)
        any_hit = any(_present(t, extract) for t in toks)
        rows.append({
            "qid": qid, "category": f.get("category", ""), "gold": str(f["gold"]),
            "extract_len": len(extract), "is_none": extract.strip().upper().startswith("NONE") or len(extract) < 8,
            "full_hit": hit, "partial_hit": any_hit, "derived": _is_derived(f),
            "structured_target": _is_structured_target(f),
        })

    if not rows:
        print("No extractions in cache yet.")
        return

    # Extraction recall is meaningful only on STRUCTURED-OBJECT golds (table/chart
    # value printed on the page). Text-passage, colour/photo-semantic, and derived
    # golds are not transcription targets and are excluded from the headline recall.
    transcribable = [r for r in rows if r["structured_target"]]
    derived = [r for r in rows if not r["structured_target"]]
    n = len(transcribable)
    full = sum(r["full_hit"] for r in transcribable)
    part = sum(r["partial_hit"] for r in transcribable)
    none = sum(r["is_none"] for r in transcribable)
    print(f"EXTRACTION RECALL ({args.extract_model})")
    print(f"  cards with extraction: {len(rows)}  | transcribable golds: {n}  | derived (excluded): {len(derived)}")
    if n:
        print(f"  full gold-value present : {full}/{n} = {full / n:.3f}   <- the SOTA-comparable recall")
        print(f"  partial (any token)     : {part}/{n} = {part / n:.3f}")
        print(f"  extraction empty/NONE   : {none}/{n} = {none / n:.3f}")
    print("\n  by category (full-hit recall, transcribable only):")
    cats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in transcribable:
        cats[r["category"]].append(r)
    for c in sorted(cats):
        cr = cats[c]
        fh = sum(x["full_hit"] for x in cr)
        nn = sum(x["is_none"] for x in cr)
        print(f"    {c:14} {len(cr):3}  recall={fh / len(cr):.3f}  (NONE/empty {nn})")

    # The misses: where a TRANSCRIBABLE gold value is NOT in the extraction (real fail).
    misses = [r for r in transcribable if not r["full_hit"]]
    print(f"\n  MISSES ({len(misses)}) — extractor did not surface the gold value:")
    for r in sorted(misses, key=lambda x: x["category"]):
        tag = "NONE/empty" if r["is_none"] else "value-absent"
        print(f"    {r['qid'].split('_')[1]:5} {r['category']:7} gold={r['gold'][:24]!r:26} [{tag}]")

    if args.out:
        args.out.write_text(json.dumps({"recall_full": full / n, "recall_partial": part / n,
            "none_rate": none / n, "n": n, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\n  Wrote {args.out}")


if __name__ == "__main__":
    main()
