"""Map time-span labels onto a recording's segment pages.

A recording is a document whose pages are its time segments. Its gold evidence
is labelled in seconds, and the current segmentation turns that into the page
numbers every page-level scorer already reads.
"""

from __future__ import annotations

import math
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
    """Sorted pages of the segments that overlap any span enough to count.

    A span too short to meet the rule on either side of a cut takes the
    segment it overlaps most, both on a tie, so a short label never maps to
    no page while it lies inside the recording."""
    pages: set[int] = set()
    for span in spans:
        counted: set[int] = set()
        best, best_pages = 0.0, []
        for seg in segments:
            overlap = min(span.end_s, seg.end_s) - max(span.start_s, seg.start_s)
            needed = min(min_overlap_s, span.end_s - span.start_s, seg.end_s - seg.start_s)
            if overlap > 0 and overlap >= needed:
                counted.add(seg.page)
            if overlap > best + 1e-9:
                best, best_pages = overlap, [seg.page]
            elif overlap > 0 and abs(overlap - best) <= 1e-9:
                best_pages.append(seg.page)
        pages |= counted or set(best_pages)
    return sorted(pages)


def random_recall_at_k(*, n_pages: int, k: int) -> float:
    """Expected recall@k of a uniformly random ranking of `n_pages` pages: each
    relevant page lands in the top k with probability k / n."""
    return min(k, n_pages) / n_pages


def random_reciprocal_rank(*, n_pages: int, n_relevant: int, depth: int | None = None) -> float:
    """Expected reciprocal rank of the first relevant page in a uniformly
    random ranking of `n_pages` pages, `n_relevant` of them relevant, counting
    only the first `depth` ranks (all of them when None)."""
    total = math.comb(n_pages, n_relevant)
    last = n_pages - n_relevant + 1 if depth is None else min(depth, n_pages - n_relevant + 1)
    return sum(math.comb(n_pages - i, n_relevant - 1) / total / i for i in range(1, last + 1))


def random_hit_at_k(*, n_pages: int, n_relevant: int, k: int) -> float:
    """Chance that a uniformly random ranking puts at least one of `n_relevant`
    relevant pages among its first `k` of `n_pages`."""
    k = min(k, n_pages)
    return 1.0 - math.comb(n_pages - n_relevant, k) / math.comb(n_pages, k)
