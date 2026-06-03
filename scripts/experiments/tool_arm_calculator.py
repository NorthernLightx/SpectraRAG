"""Calculator-tool arm: does a code-interpreter recover the arithmetic failures?

A real code-interpreter tool fixes COMPUTATION errors, not perception errors. So
this arm separates the two: the vision reader sees the gold page and must emit
(a) the operands it reads off the page and (b) a Python arithmetic expression
over them; then PYTHON evaluates the expression deterministically and we score
the computed result. Any lift over the baseline answer is the calculator doing
the math the model got wrong — but ONLY when the model read the operands right.
If the model misreads an operand, the calculator faithfully computes a wrong
answer, which is the honest ceiling: a calculator cannot fix a misread.

Scope: the arithmetic subset of the 43 post-retrieval failures (questions whose
answer is a computed number — gap/difference/average/count-then-subtract). The
subset is passed in (--qids) so the choice is explicit and auditable, not
inferred. For each, we also keep the model's own one-shot answer for contrast.

NOT authoring ground truth: gold is the human MMLongBench label; this only changes
HOW the final number is produced (deterministic eval vs the model's mental math).

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.tool_arm_calculator \
        --qids mmlb_0033_05-03-18-political-release mmlb_0035_05-03-18-political-release \
               mmlb_0053_2310.07609v1 mmlb_0056_2303.05039v2 mmlb_0138_2307.09288v2 \
        --reader-model google/gemma-4-31b-it:free \
        --out data/eval/runs/tool_arm_calculator.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import operator
import re
import sys
from pathlib import Path
from typing import Any

from scripts.experiments._openrouter_client import build_openrouter_client
from scripts.experiments.run_mmlb_qa import _chat_vision_openrouter
from scripts.experiments.score_mmlb_qa import _eval_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_CALC_SYSTEM = (
    "You answer a numeric question from a document page image. Do NOT do the "
    "arithmetic yourself. Instead:\n"
    "1. Read the raw values needed off the page (state each with its label).\n"
    "2. Write a single Python arithmetic expression over those raw numbers that "
    "computes the final answer.\n"
    'Reply with ONLY a JSON object: {"operands": "<what you read, with labels>", '
    '"expression": "<a pure Python arithmetic expression, numbers and + - * / ( ) '
    'and round() only>"}. No other text. The expression must be evaluatable as-is.'
)
_CALC_USER = "Question: {query}\n\nReturn the operands you read and the Python expression. JSON only."

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Safe arithmetic eval: numbers + - * / ( ) and round(). No names, no calls except round.
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str) -> float | None:
    """Evaluate a pure arithmetic expression (plus round()) with no names/attrs."""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "round":
            args = [ev(a) for a in n.args]
            return round(args[0], int(args[1]) if len(args) > 1 else None)
        raise ValueError(f"disallowed node: {ast.dump(n)}")

    try:
        return float(ev(node))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


async def run(args: argparse.Namespace) -> int:
    failures = {f["qid"]: f for f in json.loads(args.failures.read_text(encoding="utf-8"))}
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    client = build_openrouter_client(timeout=args.timeout)

    rows: list[dict[str, Any]] = []
    for qid in args.qids:
        f = failures.get(qid)
        if f is None:
            print(f"  {qid}: not in failures set, skipped")
            continue
        gold, fmt, question = str(f["gold"]), f["fmt"], f["query"]
        pages = [Path(p) for p in f["gold_page_pngs"] if Path(p).exists()]
        if not pages:
            print(f"  {qid}: no gold pngs, skipped")
            continue
        baseline_ans = cache.get(f"baseline::{args.reader_model}::{qid}", "")
        base_score = _eval_score(gold, baseline_ans, fmt)

        text, _, _ = await _chat_vision_openrouter(
            client, args.reader_model, _CALC_SYSTEM,
            _CALC_USER.format(query=question), pages,
            temperature=0.0, max_tokens=400,
        )
        m = _JSON_RE.search(text)
        operands = expr = ""
        computed: float | None = None
        if m:
            try:
                obj = json.loads(m.group(0))
                operands = str(obj.get("operands", ""))[:300]
                expr = str(obj.get("expression", ""))
                computed = _safe_eval(expr)
            except (ValueError, TypeError):
                pass
        calc_pred = "" if computed is None else (
            str(int(computed)) if computed == int(computed) else f"{computed:.4f}"
        )
        calc_score = _eval_score(gold, calc_pred, fmt)
        rows.append({
            "qid": qid, "gold": gold, "fmt": fmt,
            "baseline_answer": baseline_ans[:120], "baseline_score": base_score,
            "operands": operands, "expression": expr, "computed": calc_pred, "calc_score": calc_score,
            "flip": "WON" if calc_score > base_score else ("lost" if calc_score < base_score else "same"),
        })
        print(f"  {qid.split('_')[1]:6} gold={gold:9} base={base_score:.1f} calc={calc_score:.1f} [{rows[-1]['flip']:4}] expr={expr[:50]!r} -> {calc_pred!r}")

    n = len(rows)
    base_acc = sum(r["baseline_score"] for r in rows) / n if n else 0.0
    calc_acc = sum(r["calc_score"] for r in rows) / n if n else 0.0
    won = sum(1 for r in rows if r["flip"] == "WON")
    lost = sum(1 for r in rows if r["flip"] == "lost")
    print(f"\nCALCULATOR ARM  reader={args.reader_model}  n={n}")
    print(f"  baseline ACC = {base_acc:.4f}")
    print(f"  calc ACC     = {calc_acc:.4f}  (delta {calc_acc - base_acc:+.4f}; won {won}, lost {lost})")
    print("  (won = deterministic compute fixed it = read operands right, math wrong;")
    print("   no-change = misread operands/expression, which a calculator cannot fix)")

    args.out.write_text(json.dumps({"reader": args.reader_model, "rows": rows,
        "baseline_acc": base_acc, "calc_acc": calc_acc, "won": won, "lost": lost}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", type=Path, default=Path("docs/research/2026-05-29-agenda/postret_failures.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/crop_reread_cache.json"))
    ap.add_argument("--qids", nargs="+", required=True, help="the arithmetic-subset qids to test")
    ap.add_argument("--reader-model", default="google/gemma-4-31b-it:free")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/tool_arm_calculator.json"))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
