"""Ingestion scorecard — cheap, transparent, trended structural quality of the
chunked corpus.

The eval harness scores *answers*; nothing scored ingestion until now (the
gap the ADR 0018 review surfaced). This runs `extract_pages` + `chunk_pages`
only — no LLM, no RAG pipeline — so it finishes in seconds and can steer
ingestion changes early. It writes a committed JSON snapshot + a Markdown
report with per-category example chunks (the "why", not just a number), and
diffs against a prior snapshot so every ingestion change shows its delta.

Graph (entities/relations) and bib-filter precision dimensions are added by
the GraphRAG spike; this is the text/metadata structural core that ADR 0017
changed and that currently has no scoreboard.

    uv run python -m scripts.eval_ingestion --tag main
    uv run python -m scripts.eval_ingestion --tag wip --diff main
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ingestion.chunking import chunk_pages
from src.ingestion.pdf import extract_pages
from src.types import Chunk

_FRAG_CHARS = 200  # below this a chunk is too short to stand alone


def _metrics(chunks: list[Chunk], n_papers: int) -> dict[str, Any]:
    if not chunks:
        return {"papers": n_papers, "chunks": 0}
    lengths = sorted(len(c.text) for c in chunks)
    sectioned = sum(1 for c in chunks if c.section)
    cross_page = sum(1 for c in chunks if len(c.page_numbers) > 1)
    sections = {f"{c.paper_id}:{c.section}" for c in chunks if c.section}
    return {
        "papers": n_papers,
        "chunks": len(chunks),
        "chunks_per_paper": round(len(chunks) / n_papers, 1),
        "chars_mean": round(statistics.fmean(lengths), 1),
        "chars_median": lengths[len(lengths) // 2],
        "chars_p10": lengths[len(lengths) // 10],
        "fragmented_pct": round(100 * sum(x < _FRAG_CHARS for x in lengths) / len(lengths), 1),
        "section_coverage_pct": round(100 * sectioned / len(chunks), 1),
        "distinct_sections": len(sections),
        "cross_page_pct": round(100 * cross_page / len(chunks), 1),
    }


def _examples(chunks: list[Chunk]) -> dict[str, list[str]]:
    """The transparency layer: see *why* a metric moved, not just that it did."""
    short = sorted(chunks, key=lambda c: len(c.text))[:5]
    no_section = [c for c in chunks if not c.section][:5]
    return {
        "shortest_chunks (fragmentation)": [
            f"{c.chunk_id} [{len(c.text)}c] {' '.join(c.text.split())[:90]}" for c in short
        ],
        "no_section (attribution gaps)": [
            f"{c.chunk_id} {' '.join(c.text.split())[:90]}" for c in no_section
        ],
    }


def _run(papers_dir: Path) -> dict[str, Any]:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    chunks: list[Chunk] = []
    for pdf in pdfs:
        chunks.extend(chunk_pages(extract_pages(pdf.stem, pdf)))
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "papers_dir": str(papers_dir),
        "metrics": _metrics(chunks, len(pdfs)),
        "examples": _examples(chunks),
    }


def _markdown(snap: dict[str, Any], tag: str) -> str:
    m = snap["metrics"]
    lines = [f"# Ingestion scorecard — `{tag}`", "", f"_{snap['generated_at']}_", ""]
    lines += [f"- **{k}**: {v}" for k, v in m.items()]
    for title, items in snap["examples"].items():
        lines += ["", f"## {title}", *[f"- {x}" for x in items]]
    return "\n".join(lines) + "\n"


def _diff(cur: dict[str, Any], prev: dict[str, Any]) -> str:
    rows = ["", f"## Δ vs prior ({prev['generated_at']})"]
    for k, now in cur["metrics"].items():
        before = prev["metrics"].get(k)
        mark = "" if before == now or not isinstance(now, int | float) else f"  ({before} → {now})"
        rows.append(f"- {k}: {now}{mark}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--papers", type=Path, default=Path("data/papers"))
    ap.add_argument("--tag", required=True, help="snapshot name, e.g. 'main' or 'wip'")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/ingestion"))
    ap.add_argument("--diff", default=None, help="tag of a committed snapshot to diff against")
    args = ap.parse_args()

    snap = _run(args.papers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.tag}.json"
    json_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    md = _markdown(snap, args.tag)

    if args.diff:
        prior_path = args.out_dir / f"{args.diff}.json"
        if prior_path.exists():
            md += _diff(snap, json.loads(prior_path.read_text(encoding="utf-8")))
        else:
            md += f"\n(no prior snapshot '{args.diff}' to diff)\n"
    (args.out_dir / f"{args.tag}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
