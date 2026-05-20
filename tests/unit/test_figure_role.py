"""ADR 0022 — figure role classifier tests.

Pin the deterministic classifier so the gallery's default view (Figures
only) and the retrieval-side filter agree on what counts as a figure
versus a decoration.
"""

from __future__ import annotations

from src.ingestion.docling_parser import _classify_figure_role
from src.types import Bbox


def _bbox(x0: float, y0: float, x1: float, y1: float) -> Bbox:
    return Bbox(x0=x0, y0=y0, x1=x1, y1=y1)


def test_figure_caption_marks_as_figure_regardless_of_size() -> None:
    # The 906-pt² real-but-small "Figure 3: Screenshots ..." case from
    # the 2026-05-20 corpus characterisation must not be dropped.
    tiny = _bbox(100, 100, 130, 130)  # 30x30 = 900 pt², below the area cut
    role = _classify_figure_role(caption="Figure 3: Screenshots of the artifacts.", bbox=tiny)
    assert role == "figure"


def test_fig_dot_n_caption_also_passes() -> None:
    tiny = _bbox(0, 0, 30, 30)
    assert _classify_figure_role(caption="Fig. 12 a) Loss curve.", bbox=tiny) == "figure"


def test_small_uncaptioned_picture_is_decoration() -> None:
    # 13x12 = 156 pt², the email-icon case from 2604.28177v1::p1::fig4.
    small = _bbox(50, 50, 63, 62)
    assert _classify_figure_role(caption="", bbox=small) == "decoration"


def test_logo_sized_picture_below_5k_is_decoration() -> None:
    # 61x15 = 915 pt², the Microsoft logo recurring on 2604.28181v1.
    logo = _bbox(100, 100, 161, 115)
    assert _classify_figure_role(caption="", bbox=logo) == "decoration"


def test_above_threshold_uncaptioned_is_unlabeled_not_dropped() -> None:
    # 124x71 = 8804 pt², the captionless SWAP-test diagram case. The
    # paper didn't caption it as "Figure N", but it's a real diagram,
    # so the classifier keeps it as `unlabeled` (gallery hides by
    # default; retrieval can still hit it).
    real = _bbox(50, 100, 174, 171)
    assert _classify_figure_role(caption="", bbox=real) == "unlabeled"


def test_missing_bbox_is_decoration() -> None:
    # No bbox means we can't sanity-check size; treat as decoration so
    # the gallery doesn't surface it by default.
    assert _classify_figure_role(caption="", bbox=None) == "decoration"


def test_caption_first_then_size_priority() -> None:
    # Even a small uncaptioned-looking text wins if it matches Figure-N.
    tiny = _bbox(0, 0, 30, 30)
    assert _classify_figure_role(caption="figure 1", bbox=tiny) == "figure"


def test_caption_with_leading_whitespace_still_matches() -> None:
    tiny = _bbox(0, 0, 30, 30)
    assert _classify_figure_role(caption="   Figure 2: ...", bbox=tiny) == "figure"


def test_caption_not_starting_with_figure_doesnt_rescue() -> None:
    # "Source: Figure 3" mentioning the word later in the caption is not
    # the paper's own figure-label, so size still rules.
    tiny = _bbox(0, 0, 30, 30)
    assert _classify_figure_role(caption="Source: Figure 3 above", bbox=tiny) == "decoration"
