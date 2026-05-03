"""Phase 3.2 per-query routing — classify queries to text-only or hybrid (text+visual).

ADR 0008 pins the design: regex/keyword classifier emits one of five categories;
{figure, table, multi_hop} route to hybrid (RRF over text + visual at page
granularity), {factual, definitional} route to text-only. Misclassification cost
is bounded — the worst case routes a figure query through text-only, which is
the strong baseline (text @ page nDCG@5 = 0.86 per ADR 0007).
"""

from __future__ import annotations

import re
from typing import Literal

Category = Literal["table", "figure", "multi_hop", "factual", "definitional"]
RoutingPath = Literal["text", "hybrid"]

# Precedence-ordered patterns. Order matters: a query like "compare Figure 3 vs
# Figure 4" matches both `figure` and `multi_hop`; precedence picks `figure`.
# Both route to hybrid so the choice only affects the observability label.
_TABLE_RE = re.compile(r"\btable\s+\d+|\bcell\b|\brow\b|\bcolumn\b", re.IGNORECASE)
_FIGURE_RE = re.compile(
    r"\bfigure\s+\d+|\bfig\.\s*\d+|\bplot\b|\bdiagram\b|\bchart\b", re.IGNORECASE
)
_MULTIHOP_RE = re.compile(
    r"\bcompare\b|\bvs\.?\b|\bversus\b|\bdifferences?\b|\bbetween\b", re.IGNORECASE
)
# Factual = numeric span OR ≥2-char uppercase acronym. NO IGNORECASE — the
# acronym half needs case sensitivity (otherwise every word would match).
_FACTUAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b|\b[A-Z]{2,}\b")


def classify_query(text: str) -> Category:
    """Map a query string to one of the five categories per ADR 0008.

    Pure function — no I/O, no side effects, deterministic. Patterns are
    intentionally small; ADR 0008 §"Caveats" covers the trade-offs.
    """
    if _TABLE_RE.search(text):
        return "table"
    if _FIGURE_RE.search(text):
        return "figure"
    if _MULTIHOP_RE.search(text):
        return "multi_hop"
    if _FACTUAL_RE.search(text):
        return "factual"
    return "definitional"


def route_for_category(category: Category) -> RoutingPath:
    """Map a category to its dispatch destination per ADR 0008 §"Decision"."""
    if category in ("figure", "table", "multi_hop"):
        return "hybrid"
    return "text"
