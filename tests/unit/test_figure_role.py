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


def test_above_threshold_uncaptioned_is_figure() -> None:
    # 124x71 = 8804 pt², the captionless SWAP-test diagram case. The paper
    # didn't caption it as "Figure N", but a large crop is real content, so
    # the terminal area branch fails it safe to `figure` (ADR 0022 source
    # fix) rather than burying it as `unlabeled`. This is the bbox-present
    # half of the live 2604.28177v1 p13 anatomical-illustration bug.
    real = _bbox(50, 100, 174, 171)
    assert _classify_figure_role(caption="", bbox=real) == "figure"


def test_missing_bbox_is_unlabeled_not_decoration() -> None:
    # No bbox to place or measure → "unknown", not "page furniture". Defaulting
    # to `decoration` would drop a bbox-less real figure from the gallery, the
    # retrieval filter, AND the VLM captioner — a silent figure-loss path.
    # `unlabeled` keeps it retrievable; the gallery still hides it by default.
    assert _classify_figure_role(caption="", bbox=None) == "unlabeled"


def test_large_uncaptioned_unlabelled_picture_is_figure() -> None:
    # The live 2604.28177v1 p13 case: a 31764-pt² anatomical illustration
    # Docling labelled `photograph`@0.24 (below the 0.30 trust threshold)
    # whose "Figure 8:" caption the layout model failed to associate. When
    # caption recovery also misses, the terminal area branch must fail it
    # safe to `figure`, not `unlabeled`.
    big = _bbox(315.8, 338.1, 513.7, 498.6)  # the measured p13 picture bbox
    assert _classify_figure_role(caption="", bbox=big) == "figure"
    # And once recovery supplies the real "Figure 8:" caption, caption-first
    # carries it regardless of the photograph mislabel.
    assert (
        _classify_figure_role(
            caption="Figure 8: Example of a retracted paper.",
            bbox=big,
            docling_label="photograph",
            confidence=0.24,
        )
        == "figure"
    )


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


def test_table_caption_only_falls_to_area_branch() -> None:
    # A bare `Table N` caption with no classifier label is not a Figure-N
    # signal (`_FIGURE_CAPTION_RE` excludes table), so it falls through to
    # the area branch. At 100x100 = 10000 pt² (>= cut) the terminal branch
    # now returns `figure` — consistent with ADR 0022's finding that a
    # caption-only `Table N` is unreliable (3/5 such picture detections are
    # real figures with table-caption bleed) and a large crop is content.
    # A confident `table` classifier label maps to `figure` separately.
    real = _bbox(0, 0, 100, 100)
    assert _classify_figure_role(caption="Table 1. Comparison of methods.", bbox=real) == "figure"


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
    # threshold, so the label is ignored and the area heuristic decides.
    # Large area, no caption → the terminal branch fails safe to `figure`
    # (ADR 0022 source fix): a big crop is content even when both upstream
    # signals fall through.
    big_uncaptioned = _bbox(0, 0, 200, 200)
    role = _classify_figure_role(
        caption="", bbox=big_uncaptioned, docling_label="logo", confidence=0.10
    )
    assert role == "figure"


def test_docling_table_label_marks_as_figure() -> None:
    # A confident table-picture is real content: Docling's separate table
    # extractor misses tables the picture detector catches, and where both
    # fire the picture carries the human caption the table chunk lacks. Mapped
    # to `figure`, the same role the extracted table chunk gets in the gallery.
    bbox = _bbox(0, 0, 200, 200)
    role = _classify_figure_role(caption="", bbox=bbox, docling_label="table", confidence=0.99)
    assert role == "figure"


def test_unknown_docling_label_falls_through_to_heuristic() -> None:
    # A future Docling release adds a new class we haven't mapped yet —
    # don't lose the chunk, let the heuristic decide.
    tiny = _bbox(0, 0, 20, 20)
    role = _classify_figure_role(
        caption="", bbox=tiny, docling_label="hand_drawn_sketch", confidence=0.99
    )
    assert role == "decoration"  # small + no caption → decoration via heuristic
