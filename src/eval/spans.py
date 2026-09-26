"""Map time-span labels onto a recording's segment pages.

A recording is a document whose pages are its time segments. Its gold evidence
is labelled in seconds, and the current segmentation turns that into the page
numbers every page-level scorer already reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.types import MediaSegment
from src.types.eval import TimeSpan

# A label drawn a moment before a slide change must not mark the slide that
# just ended, so a segment needs this much overlap to count, unless the span
# or the segment is shorter than it.
MIN_OVERLAP_S = 1.0


def pages_for_spans(
    spans: Iterable[TimeSpan],
    segments: Sequence[MediaSegment],
    *,
    min_overlap_s: float = MIN_OVERLAP_S,
) -> list[int]:
    """Sorted pages of the segments that overlap any span enough to count."""
    pages: set[int] = set()
    for span in spans:
        for seg in segments:
            overlap = min(span.end_s, seg.end_s) - max(span.start_s, seg.start_s)
            needed = min(min_overlap_s, span.end_s - span.start_s, seg.end_s - seg.start_s)
            if overlap > 0 and overlap >= needed:
                pages.add(seg.page)
    return sorted(pages)
