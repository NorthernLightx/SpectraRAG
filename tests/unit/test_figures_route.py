"""Gallery floor for tiny decoration detections + table-picture role override.

`_is_tiny_decoration` hides glyph-sized `role=decoration` detections (12-pt
table emoji / inline icons) from the /figures gallery. It must never touch a
real figure — real figures are never `role=decoration` — and must keep
decorations large enough to be a real logo, plus any bbox-less item.

`_to_browse_item` promotes a confident table-picture baked `role=unlabeled`
(corpora ingested before `table → figure`) to `figure` at view time.
"""

from __future__ import annotations

from src.api.routes.figures import (
    FigureBrowseItem,
    Role,
    _is_tiny_decoration,
    _to_browse_item,
)
from src.types import Chunk


def _item(*, role: Role, bbox: list[float] | None) -> FigureBrowseItem:
    return FigureBrowseItem(
        chunk_id="p::p1::fig1",
        paper_id="p",
        page_number=1,
        caption="",
        page_image_url="/pages/p/p_p1.png",
        role=role,
        bbox=bbox,
    )


def test_tiny_decoration_is_hidden() -> None:
    # ~144 pt² (12x12 pt) glyph, like the AEGIS comparison-table emoji.
    assert _is_tiny_decoration(_item(role="decoration", bbox=[0.0, 0.0, 12.0, 12.0])) is True


def test_larger_decoration_is_kept() -> None:
    # ~1600 pt² — a real logo, above the floor; still browsable as a decoration.
    assert _is_tiny_decoration(_item(role="decoration", bbox=[0.0, 0.0, 40.0, 40.0])) is False


def test_small_figure_is_never_hidden() -> None:
    # Same tiny area but role=figure (e.g. a caption-rescued thumbnail) — kept.
    assert _is_tiny_decoration(_item(role="figure", bbox=[0.0, 0.0, 12.0, 12.0])) is False


def test_decoration_without_bbox_is_kept() -> None:
    assert _is_tiny_decoration(_item(role="decoration", bbox=None)) is False


def _picture_chunk(*, role: str, docling_label: str, confidence: float) -> Chunk:
    # A kind=figure picture detection with a baked role + Docling label.
    return Chunk(
        chunk_id="p::p16::fig9",
        paper_id="p",
        page_numbers=[16],
        text="A picture region.",
        metadata={
            "kind": "figure",
            "bbox": [0.0, 0.0, 200.0, 200.0],
            "role": role,
            "docling_label": docling_label,
            "docling_label_confidence": confidence,
        },
    )


def test_unlabeled_table_picture_shows_as_figure() -> None:
    # The gallery surfaces figure-vs-decoration; an "unlabeled" table picture
    # (Docling extracts tables separately) is real content → shown as figure.
    item = _to_browse_item(_picture_chunk(role="unlabeled", docling_label="table", confidence=0.99))
    assert item is not None and item.role == "figure"


def test_unlabeled_low_confidence_photograph_shows_as_figure() -> None:
    # The 2604.28177v1 p13 case: a real figure Docling labelled photograph below
    # the trust threshold and whose "Figure N" caption it failed to associate.
    # Still a real figure — not surfaced as "unlabeled".
    item = _to_browse_item(
        _picture_chunk(role="unlabeled", docling_label="photograph", confidence=0.24)
    )
    assert item is not None and item.role == "figure"


def test_decoration_is_not_promoted() -> None:
    # decoration is the one genuine "page furniture" bucket — never promoted.
    item = _to_browse_item(_picture_chunk(role="decoration", docling_label="logo", confidence=0.99))
    assert item is not None and item.role == "decoration"
