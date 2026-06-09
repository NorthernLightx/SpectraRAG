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


def _table_picture_chunk(*, role: str, docling_label: str, confidence: float) -> Chunk:
    # A picture-side detection of a table (kind=figure, Docling label "table").
    return Chunk(
        chunk_id="p::p16::fig9",
        paper_id="p",
        page_numbers=[16],
        text="Table B.2: Configurations for ImageNet class-conditional post-training.",
        metadata={
            "kind": "figure",
            "bbox": [0.0, 0.0, 200.0, 200.0],
            "role": role,
            "docling_label": docling_label,
            "docling_label_confidence": confidence,
        },
    )


def test_confident_table_picture_overrides_baked_unlabeled() -> None:
    # Pre-mapping corpora baked role="unlabeled" on a docling table-picture.
    # The view layer promotes it to "figure" so obvious tables don't surface
    # as "unlabeled" in the gallery.
    item = _to_browse_item(
        _table_picture_chunk(role="unlabeled", docling_label="table", confidence=0.99)
    )
    assert item is not None and item.role == "figure"


def test_low_confidence_table_picture_keeps_baked_role() -> None:
    # Below the trust threshold (0.30) we don't override — the baked role stands.
    item = _to_browse_item(
        _table_picture_chunk(role="unlabeled", docling_label="table", confidence=0.10)
    )
    assert item is not None and item.role == "unlabeled"


def test_non_table_unlabeled_is_untouched() -> None:
    # The override is table-specific: a genuinely uncertain non-table picture
    # (Docling "other") stays unlabeled even at high confidence.
    item = _to_browse_item(
        _table_picture_chunk(role="unlabeled", docling_label="other", confidence=0.99)
    )
    assert item is not None and item.role == "unlabeled"
