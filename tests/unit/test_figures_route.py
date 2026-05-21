"""Gallery floor for tiny decoration detections (ADR 0022 amendment).

`_is_tiny_decoration` hides glyph-sized `role=decoration` detections (12-pt
table emoji / inline icons) from the /figures gallery. It must never touch a
real figure — real figures are never `role=decoration` — and must keep
decorations large enough to be a real logo, plus any bbox-less item.
"""

from __future__ import annotations

from src.api.routes.figures import FigureBrowseItem, Role, _is_tiny_decoration


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
