"""Bet 1 (research-strategist agenda, 2026-06-01): build a HUMAN gold-audit surface.

The 2026-05-29 taxonomy estimates ~29% of post-retrieval "failures" are strict-
scoring artifacts and ~17% are bad/unprovable gold — i.e. a large share of the
measured ~0.46 ceiling is the RULER, not the model. Before any further model
work, the gold slice must be human-audited so future experiments are decidable
(every prior model lever died inside judge/scorer noise).

THE IRON RULE (docs/evals.md, eval-harvest-promote skill): the machine never
authors ground truth. This script ONLY assembles evidence for a human to judge —
it renders each gold-present failure (question, human gold, what the model
answered, and the actual gold page image(s)) into a self-contained HTML page with
a 4-way verdict control. The human adjudicates in the browser and exports
verdicts.json. A separate apply step (apply_gold_audit.py) consumes those human
verdicts. This script writes NO labels and emits NO verdict of its own.

The 4 verdicts a human assigns per query:
  - gold_correct        : the gold answer is right and the page supports it.
  - gold_wrong          : the page contradicts the gold (mislabeled, e.g. "Red"
                          for a pink element). Human supplies the corrected value.
  - gold_unprovable     : the fact needed to confirm gold is NOT on the fed page(s).
  - format_only_mismatch: model's content matches gold; only formatting differs
                          (prose-vs-list, "21%" vs "21") — a scorer artifact.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.build_gold_audit \
        --failures docs/research/2026-05-29-agenda/postret_failures.json \
        --out data/eval/audit/gold_audit.html
Then open the HTML in a browser, adjudicate, click "Download verdicts.json".
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _img_data_uri(path: Path) -> str | None:
    """Inline a PNG as a base64 data URI so the HTML is self-contained (works from
    file://, no server, no broken relative paths). None if the file is missing."""
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.standard_b64encode(path.read_bytes()).decode()


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_PAGE_CSS = """
* { box-sizing: border-box; }
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f4f4f5; color: #18181b; }
header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #e4e4e7;
         padding: 12px 20px; z-index: 10; display: flex; gap: 16px; align-items: center; }
header h1 { font-size: 16px; margin: 0; }
#progress { font-weight: 600; color: #2563eb; }
button.export { margin-left: auto; background: #16a34a; color: #fff; border: 0;
                padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; }
button.export:disabled { background: #a1a1aa; cursor: not-allowed; }
.card { background: #fff; border: 1px solid #e4e4e7; border-radius: 10px; margin: 16px 20px;
        padding: 16px; }
.card.done { border-color: #16a34a; background: #f0fdf4; }
.qid { font: 12px monospace; color: #71717a; }
.q { font-size: 16px; font-weight: 600; margin: 6px 0; }
.row { display: flex; gap: 12px; margin: 8px 0; flex-wrap: wrap; }
.box { flex: 1; min-width: 240px; padding: 8px 10px; border-radius: 6px; border: 1px solid #e4e4e7; }
.box .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #71717a; }
.box.gold { background: #fefce8; border-color: #fde047; }
.box.model { background: #eff6ff; border-color: #bfdbfe; }
.box .val { font-weight: 600; margin-top: 2px; white-space: pre-wrap; word-break: break-word; }
.pages img { max-width: 100%; border: 1px solid #d4d4d8; border-radius: 6px; margin: 6px 0;
             display: block; }
.verdicts { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.verdicts label { display: flex; align-items: center; gap: 6px; padding: 6px 10px;
                  border: 1px solid #d4d4d8; border-radius: 6px; cursor: pointer; font-size: 14px; }
.verdicts label:hover { background: #fafafa; }
.verdicts input:checked + span { font-weight: 700; }
.correction { margin-top: 8px; display: none; }
.correction.show { display: block; }
.correction input { width: 100%; padding: 6px 8px; border: 1px solid #d4d4d8; border-radius: 6px; }
.note { margin-top: 6px; }
.note input { width: 100%; padding: 6px 8px; border: 1px solid #e4e4e7; border-radius: 6px;
              font-size: 13px; }
footer { padding: 40px 20px; text-align: center; color: #71717a; }
"""

# The four human verdicts. Value strings are what apply_gold_audit.py keys on.
_VERDICTS = [
    ("gold_correct", "Gold correct"),
    ("format_only_mismatch", "Format-only mismatch (model was right)"),
    ("gold_wrong", "Gold wrong (page contradicts it)"),
    ("gold_unprovable", "Gold unprovable from page"),
]


def _card_html(i: int, total: int, rec: dict[str, Any]) -> str:
    qid = _esc(rec["qid"])
    imgs = []
    for p in rec["gold_page_pngs"]:
        uri = _img_data_uri(Path(p))
        if uri is None:
            imgs.append(f'<div style="color:#dc2626">MISSING IMAGE: {_esc(p)}</div>')
        else:
            imgs.append(f'<img src="{uri}" alt="{_esc(p)}" loading="lazy">')
    pages_html = "\n".join(imgs)

    radios = []
    for val, lbl in _VERDICTS:
        radios.append(
            f'<label><input type="radio" name="v_{qid}" value="{val}" '
            f'onchange="onVerdict(\'{qid}\', this.value)">'
            f"<span>{_esc(lbl)}</span></label>"
        )
    radios_html = "\n".join(radios)

    return f"""
<div class="card" id="card_{qid}" data-qid="{qid}">
  <div class="qid">[{i + 1}/{total}] {qid} &middot; category={_esc(rec.get("category", ""))} &middot; fmt={_esc(rec.get("fmt", ""))}</div>
  <div class="q">{_esc(rec["query"])}</div>
  <div class="row">
    <div class="box gold"><div class="lbl">Human gold</div><div class="val">{_esc(rec["gold"])}</div></div>
    <div class="box model"><div class="lbl">Model answered (scored wrong)</div><div class="val">{_esc(rec.get("model_answer", ""))}</div></div>
  </div>
  <div class="pages">{pages_html}</div>
  <div class="verdicts">{radios_html}</div>
  <div class="correction" id="corr_{qid}">
    <label class="lbl">Corrected gold value (only if "gold wrong"):</label>
    <input type="text" id="corrinput_{qid}" placeholder="what the page actually shows"
           oninput="onCorrection('{qid}', this.value)">
  </div>
  <div class="note">
    <input type="text" placeholder="optional note / page evidence"
           oninput="onNote('{qid}', this.value)">
  </div>
</div>"""


def build_html(failures: list[dict[str, Any]]) -> str:
    total = len(failures)
    cards = "\n".join(_card_html(i, total, rec) for i, rec in enumerate(failures))
    qids = json.dumps([r["qid"] for r in failures])
    # The page script holds verdicts in-memory and exports them; it NEVER fills a
    # default — an un-adjudicated query is simply absent from the export, so the
    # apply step can tell adjudicated from skipped. No machine-authored truth.
    script = """
const QIDS = __QIDS__;
const verdicts = {};   // qid -> {verdict, corrected, note}
function ensure(qid){ if(!verdicts[qid]) verdicts[qid] = {verdict:null, corrected:"", note:""}; return verdicts[qid]; }
function onVerdict(qid, v){
  ensure(qid).verdict = v;
  document.getElementById("corr_"+qid).classList.toggle("show", v === "gold_wrong");
  document.getElementById("card_"+qid).classList.add("done");
  refresh();
}
function onCorrection(qid, val){ ensure(qid).corrected = val; }
function onNote(qid, val){ ensure(qid).note = val; }
function refresh(){
  const n = Object.values(verdicts).filter(x => x.verdict).length;
  document.getElementById("progress").textContent = n + " / " + QIDS.length + " adjudicated";
  document.getElementById("exportBtn").disabled = (n === 0);
}
function exportVerdicts(){
  // Export ONLY adjudicated queries; skipped ones are omitted on purpose.
  const out = {};
  for (const qid of QIDS){ if (verdicts[qid] && verdicts[qid].verdict) out[qid] = verdicts[qid]; }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdicts.json";
  a.click();
}
""".replace("__QIDS__", qids)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold audit — {total} gold-present failures</title>
<style>{_PAGE_CSS}</style></head>
<body>
<header>
  <h1>Gold audit ({total} gold-present failures)</h1>
  <span id="progress">0 / {total} adjudicated</span>
  <button class="export" id="exportBtn" disabled onclick="exportVerdicts()">Download verdicts.json</button>
</header>
{cards}
<footer>Adjudicate each, then download verdicts.json and run apply_gold_audit.py.
The machine authors no labels — every verdict here is yours.</footer>
<script>{script}</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", type=Path, default=Path("docs/research/2026-05-29-agenda/postret_failures.json"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/audit/gold_audit.html"))
    args = ap.parse_args()

    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(failures), encoding="utf-8")
    n_imgs = sum(len(f["gold_page_pngs"]) for f in failures)
    print(f"Built {args.out} — {len(failures)} failures, {n_imgs} page images inlined.")
    print("Open it in a browser, adjudicate each, click 'Download verdicts.json',")
    print("then: .venv/Scripts/python.exe -m scripts.experiments.apply_gold_audit --verdicts verdicts.json")


if __name__ == "__main__":
    main()
