"""Render an EvalRun as JSON snapshot + readable markdown."""

from __future__ import annotations

from pathlib import Path

from src.eval.latency import latency_stats
from src.types import EvalRun


def write_run_json(run: EvalRun, path: Path) -> None:
    """Write the canonical JSON snapshot. The dashboard reads from these."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def write_run_markdown(run: EvalRun, path: Path) -> None:
    """Write a human-readable markdown report next to the JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(run), encoding="utf-8")


def render_markdown(run: EvalRun) -> str:
    """Build a markdown document from an EvalRun."""
    lines: list[str] = [
        f"# Eval Report — {run.golden_set_name} {run.golden_set_version}",
        "",
        f"- **Run ID:** `{run.run_id}`",
        f"- **Started:** {run.started_at.isoformat()}",
        f"- **Finished:** {run.finished_at.isoformat()}",
        f"- **Queries:** {len(run.per_query)}",
        "",
    ]

    if run.config:
        lines.append("## Configuration")
        lines.append("")
        for key in sorted(run.config):
            lines.append(f"- `{key}`: `{run.config[key]}`")
        lines.append("")

    in_corpus = [q for q in run.per_query if q.category != "out_of_corpus"]
    if in_corpus:
        ndcg5 = sum(q.retrieval.ndcg_at_5 for q in in_corpus) / len(in_corpus)
        recall10 = sum(q.retrieval.recall_at_10 for q in in_corpus) / len(in_corpus)
        mrr = sum(q.retrieval.mrr for q in in_corpus) / len(in_corpus)
    else:
        ndcg5 = recall10 = mrr = 0.0

    lines.extend(
        [
            "## Retrieval (in-corpus queries)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| nDCG@5 (macro) | {ndcg5:.4f} |",
            f"| recall@10 (macro) | {recall10:.4f} |",
            f"| MRR (macro) | {mrr:.4f} |",
            f"| n in-corpus queries | {len(in_corpus)} |",
            "",
        ]
    )

    citation_rates = [
        q.generation.citation_rate
        for q in run.per_query
        if q.generation is not None and q.generation.citation_rate is not None
    ]
    if citation_rates:
        mean_cr = sum(citation_rates) / len(citation_rates)
        lines.extend(
            [
                "## Generation",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| citation grounding (mean) | {mean_cr:.4f} |",
                f"| total tokens in | {sum(q.tokens_in for q in run.per_query)} |",
                f"| total tokens out | {sum(q.tokens_out for q in run.per_query)} |",
                "",
            ]
        )

    stats = latency_stats([q.latency_ms for q in run.per_query])
    lines.extend(
        [
            "## Latency",
            "",
            "| Metric | Value (ms) |",
            "|---|---|",
            f"| p50 | {stats.p50_ms:.0f} |",
            f"| p95 | {stats.p95_ms:.0f} |",
            f"| mean | {stats.mean_ms:.1f} |",
            f"| n | {stats.n} |",
            "",
        ]
    )

    lines.extend(
        [
            "## Per-Query Results",
            "",
            "| query_id | category | nDCG@5 | recall@10 | MRR | latency (ms) | cite. ground |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for query_result in run.per_query:
        cr = (
            query_result.generation.citation_rate
            if query_result.generation and query_result.generation.citation_rate is not None
            else None
        )
        cr_str = f"{cr:.3f}" if isinstance(cr, float) else "—"
        lines.append(
            f"| `{query_result.query_id}` | {query_result.category} | "
            f"{query_result.retrieval.ndcg_at_5:.3f} | "
            f"{query_result.retrieval.recall_at_10:.3f} | "
            f"{query_result.retrieval.mrr:.3f} | "
            f"{query_result.latency_ms} | {cr_str} |"
        )

    return "\n".join(lines) + "\n"
