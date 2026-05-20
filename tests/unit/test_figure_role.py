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


def test_subfigure_letter_caption_marks_as_figure() -> None:
    # 15 real figures on eval_docling_mm had this exact shape — "(a) CDF
    # of prediction errors with adaptive and static model" — and were
    # mis-tagged as unlabeled before this pattern landed.
    real = _bbox(50, 50, 161, 121)  # 11k area, above the cut
    assert _classify_figure_role(caption="(a) CDF of prediction errors.", bbox=real) == "figure"
    assert _classify_figure_role(caption="(b) Qwen3-14B with locking.", bbox=real) == "figure"


def test_letter_numbered_figure_caption() -> None:
    # "Figure C.1: Screenshot of the voting page" appendix-style numbering.
    real = _bbox(0, 0, 100, 100)
    assert (
        _classify_figure_role(caption="Figure C.1: Screenshot of voting page.", bbox=real)
        == "figure"
    )
    assert _classify_figure_role(caption="Figure F. Samples of AMD.", bbox=real) == "figure"


def test_leading_page_number_artifact_doesnt_block_match() -> None:
    # PDF extraction occasionally glues the page number onto the next
    # block: "1 Figure 9: The trade-off ...". Real-world artifact on
    # 2604.28186v1::p43::fig9 — must still classify as figure.
    real = _bbox(0, 0, 100, 100)
    assert _classify_figure_role(caption="1 Figure 9: The trade-off.", bbox=real) == "figure"


def test_table_caption_is_not_a_figure() -> None:
    # Docling has a separate `tables` extraction path. When the picture
    # detector ALSO fires on a table region, leaving it as `unlabeled`
    # avoids double-counting it as a figure. The real Table chunk lives
    # in the tables list (different `kind`).
    real = _bbox(0, 0, 100, 100)
    assert (
        _classify_figure_role(caption="Table 1. Comparison of methods.", bbox=real) == "unlabeled"
    )


def test_random_subscript_letter_word_doesnt_match_subfig() -> None:
    # "(a great experiment) ..." has parens-letter shape but the closing
    # paren is far away — \([a-z]\) requires exactly one letter inside.
    tiny = _bbox(0, 0, 30, 30)
    assert _classify_figure_role(caption="(a great experiment shows)", bbox=tiny) == "decoration"


# ----- Docling classifier integration --------------------------------


def test_docling_label_logo_overrides_size_heuristic() -> None:
    # A 100x100 (10000-pt^2) picture would pass the area test as
    # `unlabeled`, but if Docling says it's a logo at high confidence
    # we trust the classifier.
    bbox = _bbox(0, 0, 100, 100)
    role = _classify_figure_role(caption="", bbox=bbox, docling_label="logo", confidence=0.99)
    assert role == "decoration"


def test_paper_caption_beats_docling_logo_mislabel() -> None:
    # The small "Figure 3: Screenshots of the artifacts ..." case from
    # 2604.28181v1 — Docling's visual classifier confidently calls a
    # 906-pt^2 thumbnail a logo, but the paper's own caption says
    # otherwise. The paper wins.
    tiny = _bbox(0, 0, 30, 30)
    role = _classify_figure_role(
        caption="Figure 3: Screenshots of the artifacts.",
        bbox=tiny,
        docling_label="logo",
        confidence=1.00,
    )
    assert role == "figure"


def test_docling_label_bar_chart_marks_as_figure() -> None:
    bbox = _bbox(0, 0, 100, 100)
    role = _classify_figure_role(caption="", bbox=bbox, docling_label="bar_chart", confidence=0.95)
    assert role == "figure"


def test_docling_label_below_confidence_falls_back_to_heuristic() -> None:
    # Classifier said "logo" at 0.10 confidence — below the trust
    # threshold; the heuristic (large area, no caption) wins and we
    # keep it as `unlabeled`.
    big_uncaptioned = _bbox(0, 0, 200, 200)
    role = _classify_figure_role(
        caption="", bbox=big_uncaptioned, docling_label="logo", confidence=0.10
    )
    assert role == "unlabeled"


def test_docling_table_label_stays_unlabeled() -> None:
    # Docling extracts tables via a separate model; the picture-
    # detector firing on a table region is a duplicate that we
    # deliberately don't promote to `figure`.
    bbox = _bbox(0, 0, 200, 200)
    role = _classify_figure_role(caption="", bbox=bbox, docling_label="table", confidence=0.99)
    assert role == "unlabeled"


def test_unknown_docling_label_falls_through_to_heuristic() -> None:
    # A future Docling release adds a new class we haven't mapped yet —
    # don't lose the chunk, let the heuristic decide.
    tiny = _bbox(0, 0, 20, 20)
    role = _classify_figure_role(
        caption="", bbox=tiny, docling_label="hand_drawn_sketch", confidence=0.99
    )
    assert role == "decoration"  # small + no caption → decoration via heuristic
