"""Recovering MCIF's human answer locations from its short-form clips.

Each answerable short-form question's clip is the segment, or the joined run of
segments, that overlaps the annotators' Answer Start and Answer End. Finding
the clip inside its talk's audio gives that location back in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.locate_mcif_spans import locate_clip

RATE = 1000  # a low rate keeps the synthetic signals small


def _talk(seconds: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(seconds * RATE).astype(np.float32)


def test_a_clip_cut_from_the_talk_is_found_at_its_offset() -> None:
    talk = _talk(300)
    clip = talk[120 * RATE : 131 * RATE]
    span = locate_clip(clip, talk, RATE)
    assert span is not None
    assert span.start_s == pytest.approx(120.0, abs=0.01)
    assert span.end_s == pytest.approx(131.0, abs=0.01)


def test_a_joined_clip_spans_from_its_first_to_its_last_segment() -> None:
    # Two consecutive segments with a pause between them, joined without it.
    talk = _talk(300)
    clip = np.concatenate([talk[40 * RATE : 52 * RATE], talk[55 * RATE : 63 * RATE]])
    span = locate_clip(clip, talk, RATE)
    assert span is not None
    assert span.start_s == pytest.approx(40.0, abs=0.01)
    assert span.end_s == pytest.approx(63.0, abs=0.01)


def test_a_clip_that_is_not_in_the_talk_is_rejected() -> None:
    talk = _talk(300)
    stranger = np.random.default_rng(1).standard_normal(11 * RATE).astype(np.float32)
    assert locate_clip(stranger, talk, RATE) is None


def test_a_rescaled_copy_is_still_found() -> None:
    # A different gain or a lossy re-encode must not break the match.
    talk = _talk(300)
    clip = (0.5 * talk[200 * RATE : 212 * RATE] + 0.01 * _talk(12)).astype(np.float32)
    span = locate_clip(clip, talk, RATE)
    assert span is not None and span.start_s == pytest.approx(200.0, abs=0.01)
