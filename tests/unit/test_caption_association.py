"""ADR 0022 — caption-recovery tests for `_associate_caption`.

Docling's layout model sometimes fails to link a picture to its on-page
"Figure N:" caption, leaving `pic.caption_text` empty so the caption-first
classifier rule can't fire (live bug: 2604.28177v1 p13). `_associate_caption`
recovers the caption from the page's text items by bbox proximity. These pin
the band geometry: below-preferred, horizontal-overlap required, nearest wins,
gap-capped.

All bboxes are project TOP-LEFT (`Bbox`): y grows downward, so "below the
picture" is larger y. The helper is pure — it takes already-flipped
`(text, Bbox)` pairs, so no Docling objects are needed here.
"""

from __future__ import annotations

from src.ingestion.docling_parser import _associate_caption
from src.types import Bbox


def _bbox(x0: float, y0: float, x1: float, y1: float) -> Bbox:
    return Bbox(x0=x0, y0=y0, x1=x1, y1=y1)


# The measured 2604.28177v1 p13 picture bbox (flipped TOP-LEFT), from the
# Docling geometry probe. Reused so these tests anchor the real bug.
_PIC = _bbox(315.8, 338.1, 513.7, 498.6)


def test_figure_caption_directly_below_is_recovered() -> None:
    # The real p13 caption sits 12.4 pt under the picture with wide
    # horizontal overlap — exactly the dropped-edge case.
    cap = _bbox(306.0, 511.0, 526.0, 543.0)
    items = [("Figure 8: Example of a retracted paper.", cap)]
    assert _associate_caption(_PIC, items) == "Figure 8: Example of a retracted paper."


def test_caption_above_is_recovered_when_no_below() -> None:
    # Above-match is the fallback (mostly tables captioned on top). Bottom
    # edge of the caption is just above the picture's top edge.
    cap = _bbox(320.0, 318.0, 510.0, 333.0)  # y1=333 < pic.y0=338.1
    items = [("Figure 8: Example of a retracted paper.", cap)]
    assert _associate_caption(_PIC, items) == "Figure 8: Example of a retracted paper."


def test_below_is_preferred_over_above_when_both_exist() -> None:
    above = _bbox(320.0, 318.0, 510.0, 333.0)
    below = _bbox(306.0, 511.0, 526.0, 543.0)
    items = [
        ("Figure 8: caption above the picture.", above),
        ("Figure 8: caption below the picture.", below),
    ]
    # Even though the above caption is geometrically closer (gap ~5 pt vs
    # ~12 pt), below wins by rule.
    assert _associate_caption(_PIC, items) == "Figure 8: caption below the picture."


def test_table_caption_below_is_recovered() -> None:
    # `Table N` is a valid primary-caption shape for recovery (the helper's
    # pattern is broader than the figure-only classifier regex).
    cap = _bbox(306.0, 511.0, 526.0, 543.0)
    items = [("Table 2: Hyperparameter grid.", cap)]
    assert _associate_caption(_PIC, items) == "Table 2: Hyperparameter grid."


def test_no_horizontal_overlap_returns_none() -> None:
    # A caption-shaped line in the *next column* (no x-overlap) is not this
    # figure's caption.
    cap = _bbox(40.0, 511.0, 250.0, 543.0)  # entirely left of pic.x0=315.8
    items = [("Figure 8: a neighbour-column caption.", cap)]
    assert _associate_caption(_PIC, items) is None


def test_far_below_beyond_band_returns_none() -> None:
    # Overlapping horizontally but well past the vertical band — belongs to
    # a different figure further down the column.
    cap = _bbox(306.0, 700.0, 526.0, 732.0)  # gap = 700 - 498.6 = 201 pt
    items = [("Figure 9: a different figure's caption.", cap)]
    assert _associate_caption(_PIC, items) is None


def test_non_caption_text_in_band_is_ignored() -> None:
    # The real p13 page has "Notably, as shown in Figures 8 and 9, ..."
    # above the picture. It mentions "Figures 8" but doesn't *start* with a
    # primary-caption token, so the pattern anchor rejects it.
    prose = _bbox(306.0, 295.0, 525.0, 318.0)
    items = [("Notably, as shown in Figures 8 and 9, only a few cases.", prose)]
    assert _associate_caption(_PIC, items) is None


def test_nearest_below_wins_among_several() -> None:
    near = _bbox(306.0, 511.0, 526.0, 543.0)  # gap ~12 pt
    far = _bbox(306.0, 560.0, 526.0, 592.0)  # gap ~61 pt, still in band
    items = [
        ("Figure 8: the far caption.", far),
        ("Figure 8: the near caption.", near),
    ]
    assert _associate_caption(_PIC, items) == "Figure 8: the near caption."


def test_empty_text_items_returns_none() -> None:
    assert _associate_caption(_PIC, []) is None
