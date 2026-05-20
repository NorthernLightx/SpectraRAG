"""docling_chunker pure helpers (ADR 0021).

The full `chunk_with_docling` integration is exercised by the eval run
(`baseline-docling-text.json`); these unit tests pin the small pure
helpers that decide which blocks make it into a chunk and how bboxes
are aggregated.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.ingestion.docling_chunker import (
    _accept_block,
    _bbox_inside_any,
    _union_bbox,
)
from src.types import Bbox


def _bbox(x0: float, y0: float, x1: float, y1: float) -> Bbox:
    return Bbox(x0=x0, y0=y0, x1=x1, y1=y1)


def _item(label: str, text: str, page: int = 1, bbox_raw: Any | None = None) -> Any:
    """Duck-typed Docling text item with `label`, `text`, and `prov[0]`."""
    prov = SimpleNamespace(page_no=page, bbox=bbox_raw)
    return SimpleNamespace(label=label, text=text, prov=[prov])


def _raw_bbox_bottomleft(left: float, top: float, right: float, bottom: float) -> Any:
    """Docling's BoundingBox is BOTTOMLEFT-origin: y=0 is page bottom, so
    `t > b`. Returned as a duck-typed object since the project doesn't
    depend on the docling types in tests."""
    return SimpleNamespace(l=left, t=top, r=right, b=bottom)


# ----- _bbox_inside_any --------------------------------------------------


def test_bbox_inside_any_returns_true_for_full_containment() -> None:
    inner = _bbox(20, 20, 50, 50)
    outer = _bbox(0, 0, 100, 100)
    assert _bbox_inside_any(inner, [outer])


def test_bbox_inside_any_returns_false_when_disjoint() -> None:
    inner = _bbox(0, 0, 10, 10)
    outer = _bbox(100, 100, 200, 200)
    assert not _bbox_inside_any(inner, [outer])


def test_bbox_inside_any_returns_false_for_partial_overlap_below_threshold() -> None:
    # Inner has 100x100 area; intersection is 20x20 = 400 → 4% overlap.
    inner = _bbox(0, 0, 100, 100)
    outer = _bbox(80, 80, 200, 200)
    assert not _bbox_inside_any(inner, [outer])


def test_bbox_inside_any_returns_true_at_threshold_overlap() -> None:
    # Inner 100x100 = 10000; intersection 80x100 = 8000 → 80%, just at threshold.
    inner = _bbox(20, 0, 120, 100)
    outer = _bbox(0, 0, 100, 100)
    assert _bbox_inside_any(inner, [outer])


# ----- _union_bbox -------------------------------------------------------


def test_union_bbox_spans_all_inputs() -> None:
    union = _union_bbox([_bbox(10, 10, 30, 30), _bbox(20, 20, 50, 60)])
    assert union is not None
    assert (union.x0, union.y0, union.x1, union.y1) == (10, 10, 50, 60)


def test_union_bbox_empty_returns_none() -> None:
    assert _union_bbox([]) is None


# ----- _accept_block -----------------------------------------------------


@pytest.fixture
def heights() -> dict[int, float]:
    return {1: 792.0}


def test_accept_block_emits_normal_text(heights: dict[int, float]) -> None:
    item = _item("text", "Hello world", bbox_raw=_raw_bbox_bottomleft(50, 700, 250, 600))
    block = _accept_block(item, heights=heights, fig_tab_by_page={})
    assert block is not None
    assert block.label == "text"
    # BOTTOMLEFT 700 (top) → TOP-LEFT y0 = 792 - 700 = 92.
    assert block.bbox is not None and block.bbox.y0 == pytest.approx(92.0)


def test_accept_block_drops_page_header(heights: dict[int, float]) -> None:
    item = _item(
        "page_header", "Preprint. Under review.", bbox_raw=_raw_bbox_bottomleft(50, 780, 200, 760)
    )
    assert _accept_block(item, heights=heights, fig_tab_by_page={}) is None


def test_accept_block_drops_page_footer(heights: dict[int, float]) -> None:
    item = _item("page_footer", "1", bbox_raw=_raw_bbox_bottomleft(50, 30, 60, 20))
    assert _accept_block(item, heights=heights, fig_tab_by_page={}) is None


def test_accept_block_drops_caption(heights: dict[int, float]) -> None:
    # Captions belong to figures/tables (figure_to_chunk owns them).
    item = _item("caption", "Figure 1: ...", bbox_raw=_raw_bbox_bottomleft(50, 500, 250, 480))
    assert _accept_block(item, heights=heights, fig_tab_by_page={}) is None


def test_accept_block_drops_text_inside_figure_bbox(heights: dict[int, float]) -> None:
    # Figure on page 1 covers a region; an axis-tick text inside it should
    # be excluded from chunkable body.
    fig_bbox = _bbox(50, 50, 250, 250)
    # axis-tick block at TOP-LEFT (60,60)-(70,70) — fully inside fig.
    # BOTTOMLEFT for (60,60)-(70,70) given page_height=792: t=732, b=722.
    item = _item("text", "64", bbox_raw=_raw_bbox_bottomleft(60, 732, 70, 722))
    assert _accept_block(item, heights=heights, fig_tab_by_page={1: [fig_bbox]}) is None


def test_accept_block_drops_too_short_text(heights: dict[int, float]) -> None:
    item = _item("text", "a", bbox_raw=_raw_bbox_bottomleft(50, 100, 60, 90))
    assert _accept_block(item, heights=heights, fig_tab_by_page={}) is None


def test_accept_block_drops_no_prov(heights: dict[int, float]) -> None:
    item = SimpleNamespace(label="text", text="real text", prov=[])
    assert _accept_block(item, heights=heights, fig_tab_by_page={}) is None
