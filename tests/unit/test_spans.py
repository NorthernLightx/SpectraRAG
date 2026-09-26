"""Time-span labels -> segment pages: the overlap rule that scores recorded talks.

Gold spans are stored in seconds so a re-segmentation never invalidates a label;
this rule decides which of the current segments count as relevant pages.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.eval.spans import pages_for_spans
from src.types import MediaSegment
from src.types.eval import GoldenQuery, TimeSpan

# Three slides: 0-20 s, 20-50 s, 50-60 s.
SEGMENTS = [
    MediaSegment(page=1, start_s=0.0, end_s=20.0),
    MediaSegment(page=2, start_s=20.0, end_s=50.0),
    MediaSegment(page=3, start_s=50.0, end_s=60.0),
]


def test_span_inside_one_segment_maps_to_that_page() -> None:
    assert pages_for_spans([TimeSpan(start_s=25.0, end_s=40.0)], SEGMENTS) == [2]


def test_span_straddling_a_cut_maps_to_both_pages() -> None:
    assert pages_for_spans([TimeSpan(start_s=15.0, end_s=30.0)], SEGMENTS) == [1, 2]


def test_sliver_overlap_at_a_cut_does_not_count() -> None:
    # A label drawn half a second early must not mark the previous slide.
    assert pages_for_spans([TimeSpan(start_s=19.5, end_s=35.0)], SEGMENTS) == [2]


def test_short_span_fully_inside_a_segment_counts() -> None:
    assert pages_for_spans([TimeSpan(start_s=52.0, end_s=52.4)], SEGMENTS) == [3]


def test_several_spans_give_sorted_unique_pages() -> None:
    spans = [TimeSpan(start_s=52.0, end_s=58.0), TimeSpan(start_s=2.0, end_s=10.0)]
    assert pages_for_spans(spans + spans, SEGMENTS) == [1, 3]


def test_no_spans_no_pages() -> None:
    assert pages_for_spans([], SEGMENTS) == []


def test_time_span_rejects_empty_or_reversed_range() -> None:
    with pytest.raises(ValidationError):
        TimeSpan(start_s=10.0, end_s=10.0)
    with pytest.raises(ValidationError):
        TimeSpan(start_s=-1.0, end_s=5.0)


def test_golden_query_reads_spans_from_yaml_shape() -> None:
    q = GoldenQuery.model_validate(
        {
            "query_id": "mcif_QA_13_135",
            "text": "What are the main data sources for language models?",
            "paper_id": "ICWfTnUMio",
            "category": "factual",
            "relevant_spans": [{"start_s": 12.0, "end_s": 31.5}],
        }
    )
    assert q.relevant_spans == [TimeSpan(start_s=12.0, end_s=31.5)]
    assert GoldenQuery.model_validate({**q.model_dump(), "relevant_spans": []}).relevant_spans == []
