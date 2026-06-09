"""ADR 0022 regression guard — a captioned figure is never surfaced as unknown.

The bug this pins (ADR 0022, 2026-06-09 amendments): a real captioned picture
(`2604.28177v1` p13, "Figure 8: ...") was stored `role=unlabeled` and shown that
way — a figure hidden behind the gallery's "unknown" bucket.

The STRUCTURAL invariant — not a labelled golden set, so the
machine-never-authors-truth rule is intact:

    A picture chunk whose caption matches a primary `Figure N` / `Fig. N` /
    `Table N` / `Tab. N` label must never be *surfaced* as `role=unlabeled`.

Surfaced means the role `figures._to_browse_item` returns — what the gallery
shows the user — which is the layer the bug appeared at. The *stored* role
legitimately keeps the 3-way split (the retrieval filter still distinguishes
`decoration`, and `unlabeled` stays a "kept but uncaptioned" marker), so the
invariant is asserted post-view, not on `chunk.metadata["role"]`.

This drives the real `_to_browse_item` and the caption-first arm of
`_classify_figure_role`. It is deliberately decoupled from the area-heuristic
terminal branch of the classifier (whose default is an ingestion-side choice):
caption-first fires before that branch, so the guard holds whichever way the
fallback defaults.
"""

from __future__ import annotations

import re

import pytest

from src.api.routes.figures import _to_browse_item
from src.ingestion.docling_parser import _classify_figure_role
from src.types import Bbox, Chunk

# The invariant's caption pattern (ADR 0022). Defined here, not imported, so the
# guard pins the contract independently of any ingestion-side regex constant —
# the rule is the spec, not a shared implementation detail.
_INVARIANT_CAPTION_RE = re.compile(
    r"^\s*(?:\d+\s+)?(?:figure|fig\.?|table|tab\.?)\s+[A-Z0-9]", re.IGNORECASE
)

# Captions that MUST be read as a primary figure/table label.
_CAPTIONS_THAT_MATCH = [
    "Figure 8: Anatomy of the reproductive system.",  # the p13 shape
    "figure 1 overview",  # lower-case, no colon
    "Fig. 12 a) Loss curve.",
    "Table B.2: Configurations for ImageNet post-training.",  # the live rebake case
    "Tab. 3 Hyperparameters.",
    "1 Figure 9: The trade-off.",  # leading page-number OCR artifact
]
# Captions that must NOT trip the invariant (so the guard can't pass vacuously
# by matching everything).
_CAPTIONS_THAT_DONT_MATCH = [
    "This is an anatomical illustration of a male mouse.",  # p13's actual VLM caption
    "Source: Figure 3 above",  # 'Figure' not at the start
    "[2604.28177v1::p13::fig12]",  # id-stub placeholder
    "(a) CDF of prediction errors.",  # subfigure shape — not a *primary* label
]


def test_invariant_regex_separates_primary_captions_from_the_rest() -> None:
    # Anchors the spec: the pattern fires on primary labels and nothing else.
    # Without this a buggy regex could make every assertion below vacuous.
    for cap in _CAPTIONS_THAT_MATCH:
        assert _INVARIANT_CAPTION_RE.match(cap), f"should match: {cap!r}"
    for cap in _CAPTIONS_THAT_DONT_MATCH:
        assert not _INVARIANT_CAPTION_RE.match(cap), f"should not match: {cap!r}"


def _captioned_picture(*, caption: str, baked_role: str | None, bbox: list[float] | None) -> Chunk:
    """A kind=figure picture chunk whose primary text IS the caption.

    `baked_role` is what a prior ingest stored in metadata (the bug shape is a
    baked ``unlabeled``); ``None`` omits the field so the API's `_derive_role`
    migration cushion runs instead.
    """
    metadata: dict[str, object] = {"kind": "figure"}
    if bbox is not None:
        metadata["bbox"] = bbox
    if baked_role is not None:
        metadata["role"] = baked_role
    return Chunk(
        chunk_id="2604.28177v1::p13::fig12",
        paper_id="2604.28177v1",
        page_numbers=[13],
        text=caption,
        metadata=metadata,
    )


@pytest.mark.parametrize("caption", _CAPTIONS_THAT_MATCH)
@pytest.mark.parametrize("baked_role", ["unlabeled", None, "figure"])
def test_captioned_picture_never_surfaces_unlabeled(caption: str, baked_role: str | None) -> None:
    """The core guard: through the real view path, a caption-matching picture is
    never surfaced as ``unlabeled`` — regardless of the role a stale bake stored
    (``unlabeled`` is the exact bug) or whether the field is absent."""
    # Force the precondition that made this a regression: the caption matches.
    assert _INVARIANT_CAPTION_RE.match(caption)
    chunk = _captioned_picture(caption=caption, baked_role=baked_role, bbox=[0.0, 0.0, 80.0, 80.0])
    item = _to_browse_item(chunk)
    assert item is not None
    assert item.role != "unlabeled", (
        f"captioned figure surfaced as unlabeled (baked={baked_role!r}, caption={caption!r})"
    )


def test_bbox_less_captioned_picture_never_surfaces_unlabeled() -> None:
    # The defended silent-loss path: a real figure that lost its bbox. The area
    # heuristic can't fire (nothing to measure), so only caption-first +
    # gallery collapse keep it visible.
    chunk = _captioned_picture(
        caption="Figure 8: Anatomy of the reproductive system.", baked_role="unlabeled", bbox=None
    )
    item = _to_browse_item(chunk)
    assert item is not None
    assert item.role != "unlabeled"


def test_classifier_caption_first_beats_a_would_be_unlabeled() -> None:
    """Pin the ingestion-side arm directly: caption-first returns ``figure`` for
    a tiny crop a confident non-figure label would otherwise bucket elsewhere —
    so a fresh ingest of a captioned picture never *bakes* ``unlabeled``.

    Driven through caption-first only; independent of the area-heuristic default,
    which is an ingestion-owned choice in flux.
    """
    tiny = Bbox(x0=0, y0=0, x1=30, y1=30)  # 900 pt², below the 5000 area cut
    assert _classify_figure_role(caption="Figure 8: Anatomy.", bbox=tiny) == "figure"
    # Even with a confident competing visual label, the paper's `Figure N`
    # caption wins. (The classifier's caption-rescue is `Figure`/`Fig.` only —
    # `Table N` caption text is deliberately not a rescue signal here, per ADR
    # 0022; the broader table/tab half of the invariant is enforced at the
    # surfaced `_to_browse_item` layer above, not in this ingestion arm.)
    assert (
        _classify_figure_role(
            caption="Figure 8: Anatomy.", bbox=tiny, docling_label="logo", confidence=0.99
        )
        == "figure"
    )


def test_uncaptioned_decoration_still_allowed_to_surface_non_figure() -> None:
    # Negative control: the invariant only protects *captioned* pictures. A
    # genuine glyph (no caption, sub-threshold) is free to be decoration — the
    # guard must not over-reach and force everything to figure.
    chunk = _captioned_picture(caption="", baked_role="decoration", bbox=[0.0, 0.0, 12.0, 12.0])
    item = _to_browse_item(chunk)
    assert item is not None
    assert item.role == "decoration"
