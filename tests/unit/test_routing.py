"""Phase 3.2 routing — classify_query precedence + Query.force_route field per ADR 0008."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rag.retrievers.routing import Category, classify_query
from src.types import Query


@pytest.mark.parametrize(
    "text,expected",
    [
        # ---- table (precedence 1) ----
        ("What is the value in Table 4?", "table"),
        ("Show me the cell at row 3 column 2", "table"),
        ("Which row of the dataset has the highest score?", "table"),
        # ---- figure (precedence 2) ----
        ("Show Figure 3 architecture", "figure"),
        ("Look at Fig. 5 in the appendix", "figure"),
        ("What does the chart show?", "figure"),
        ("Describe the diagram on page 7", "figure"),
        # ---- multi_hop (precedence 3) ----
        ("How does method X compare to method Y?", "multi_hop"),
        ("Method A versus Method B", "multi_hop"),
        ("differences between approach 1 and approach 2", "multi_hop"),
        ("Choose between option A or B", "multi_hop"),
        # ---- factual (precedence 4 — numeric span OR ≥2-char acronym) ----
        ("What is the FID score?", "factual"),  # FID = acronym
        ("model achieves 0.85 accuracy", "factual"),  # numeric
        # ---- definitional (precedence 5 — default) ----
        ("What does the model do?", "definitional"),
        ("Explain the methodology", "definitional"),
        ("describe the approach in plain terms", "definitional"),
        # ---- precedence wins ----
        # figure beats multi_hop when both signals present (ADR 0008 §"Classifier")
        ("Compare Figure 3 vs Figure 4", "figure"),
        # table beats multi_hop when both signals present
        ("Compare data in Table 2", "table"),
        # table beats figure if both Table and Figure tokens are present
        ("Figure 3 reproduces Table 4 metrics", "table"),
    ],
)
def test_classify_query(text: str, expected: Category) -> None:
    assert classify_query(text) == expected


def test_classify_query_handles_empty_string() -> None:
    """Edge case: empty input falls through to the default. The Query model
    rejects empty text upstream, but classify_query is a pure function and
    should be safe to call regardless."""
    assert classify_query("") == "definitional"


def test_query_force_route_defaults_to_none() -> None:
    q = Query(text="hello world", top_k=5)
    assert q.force_route is None


def test_query_force_route_accepts_text() -> None:
    q = Query(text="hello", top_k=5, force_route="text")
    assert q.force_route == "text"


def test_query_force_route_accepts_hybrid() -> None:
    q = Query(text="hello", top_k=5, force_route="hybrid")
    assert q.force_route == "hybrid"


def test_query_force_route_rejects_invalid_literal() -> None:
    """Pydantic Literal validation should reject anything outside {text, hybrid, None}."""
    with pytest.raises(ValidationError):
        Query(text="hello", top_k=5, force_route="visual")
