"""harvest/promote: reference-free flag heuristics + the human-gate validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.harvest_candidates import _flag_reasons, _is_refusal, _to_candidate
from scripts.promote_candidates import NotLabeledError, _validate_candidate


def _pq(
    *,
    category: str = "factual",
    answer: str | None = "The value is 42 [p::p1::c1].",
    faith: float | None = None,
    cp: float | None = None,
    retrieved: list[str] | None = None,
) -> dict[str, object]:
    gen: dict[str, object] = {}
    if faith is not None:
        gen["faithfulness"] = faith
    if cp is not None:
        gen["context_precision"] = cp
    return {
        "query_id": "q1",
        "category": category,
        "text": "What is the value?",
        "answer_text": answer,
        "retrieved_chunk_ids": ["p::p1::c1"] if retrieved is None else retrieved,
        "generation": gen or None,
    }


def test_is_refusal_matches_phrases() -> None:
    assert _is_refusal("Not stated in the provided context.")
    assert _is_refusal("I cannot answer this question from the corpus.")
    assert not _is_refusal("The answer is 42.")
    assert not _is_refusal(None)


def test_low_judged_metric_flagged() -> None:
    assert "low_faithfulness=0.10" in _flag_reasons(_pq(faith=0.1))
    assert _flag_reasons(_pq(cp=0.2)) == ["low_context_precision=0.20"]


def test_false_refusal_flagged() -> None:
    # answerable category, but the model refused
    assert "false_refusal" in _flag_reasons(
        _pq(category="factual", answer="Not stated in the provided context.")
    )


def test_missing_refusal_flagged() -> None:
    # OOC should have refused, but the model answered
    assert "missing_refusal" in _flag_reasons(_pq(category="out_of_corpus", answer="It is 42."))


def test_empty_retrieval_flagged() -> None:
    assert "empty_retrieval" in _flag_reasons(_pq(retrieved=[]))


def test_clean_case_not_flagged() -> None:
    assert _flag_reasons(_pq(faith=0.95, cp=0.9)) == []


def test_candidate_stub_has_blank_truth_fields() -> None:
    c = _to_candidate(_pq(faith=0.1), run_id="r1")
    assert c["category"] == "TODO" and c["paper_id"] == "TODO"
    assert c["expected_facts"] == [] and c["relevant_chunk_ids"] == []
    assert "low_faithfulness" in c["note"]


def test_validate_rejects_unlabeled_stub() -> None:
    stub = _to_candidate(_pq(faith=0.1), run_id="r1")
    # category="TODO" is not a valid QueryCategory → ValidationError
    with pytest.raises(ValidationError):
        _validate_candidate(stub)


def test_validate_rejects_valid_category_but_empty_truth() -> None:
    half = {
        "query_id": "cand_q1",
        "text": "What is the value?",
        "paper_id": "2604.22753v1",
        "category": "factual",
        "relevant_chunk_ids": [],
        "relevant_pages": [],
        "expected_facts": [],
    }
    with pytest.raises(NotLabeledError):
        _validate_candidate(half)


def test_validate_accepts_fully_labeled() -> None:
    good = {
        "query_id": "cand_q1",
        "text": "What is the value?",
        "paper_id": "2604.22753v1",
        "category": "factual",
        "relevant_chunk_ids": ["2604.22753v1::p1::c1"],
        "relevant_pages": [],
        "expected_facts": ["The value is 42."],
    }
    q = _validate_candidate(good)
    assert q.query_id == "cand_q1"
    assert q.expected_facts == ["The value is 42."]
