"""VisualRetriever: query embedding + MaxSim scoring + ranking via stubs.

The real ColQwen2 model is too big to load in unit tests; we substitute a
stub model + processor that mimics the colpali-engine API surface. This
keeps the test fast (<1 s) and proves the retriever's *plumbing* is right
(top-K ranking, score parsing, batched score path, return shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.rag.retrievers.visual import VisualRetriever
from src.types import Query


class _BatchLike:
    """Mimics transformers' `BatchEncoding` enough for `batch.to(device)` plus
    `**batch` dict unpacking in `model(**batch)` — the two operations the
    retriever performs after `process_queries(...)`."""

    def to(self, _device: str) -> _BatchLike:
        return self

    def keys(self) -> list[str]:  # `**self` requires keys() + __getitem__
        return []

    def __getitem__(self, _key: str) -> Any:
        raise KeyError(_key)


@dataclass
class _StubProcessor:
    """Mimics ColQwen2Processor.score_multi_vector with a fixed score table.

    score_multi_vector(qs, ps) gets called with qs=[1, n_q, dim] and ps=list of
    [n_p, dim] tensors. Our stub ignores the actual content and returns a
    pre-canned [1, len(ps)] score row indexed by the chunk_id list passed in
    via `chunk_ids` (set by the caller before each retrieve).
    """

    chunk_ids: list[str]
    score_by_chunk: dict[str, float]

    def process_queries(self, queries: list[str]) -> _BatchLike:
        # Real ColQwen2Processor returns a BatchEncoding with `.to(device)`
        # and dict-style unpacking via `**batch`. The stub must support both.
        return _BatchLike()

    def score_multi_vector(self, query_embed: Any, page_tensors: Any) -> torch.Tensor:
        # Build the canned [1, B_p] row in the order the retriever passed pages.
        row = torch.tensor(
            [[self.score_by_chunk.get(cid, 0.0) for cid in self.chunk_ids]],
            dtype=torch.float32,
        )
        return row


class _StubModel:
    """Returns a placeholder query embedding tensor; not actually used by the
    stub processor's score function (it uses score_by_chunk instead)."""

    def __call__(self, **_kwargs: Any) -> torch.Tensor:
        return torch.zeros((1, 8, 16))


def _page_no(chunk_id: str) -> int:
    # chunk_id format: `<paper>::p<N>::page` — find the second `::` segment.
    parts = chunk_id.split("::")
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            return int(part[1:])
    raise AssertionError(f"can't parse page from {chunk_id!r}")


def _make_retriever(scores: dict[str, float]) -> VisualRetriever:
    page_embeds = {cid: torch.zeros((20, 16)) for cid in scores}
    paper_id = next(iter(scores)).split("::", 1)[0]
    page_meta = {cid: (paper_id, _page_no(cid)) for cid in scores}
    proc = _StubProcessor(chunk_ids=list(page_embeds.keys()), score_by_chunk=scores)
    return VisualRetriever(
        model=_StubModel(),
        processor=proc,
        page_embeds=page_embeds,
        page_meta=page_meta,
        device="cpu",
    )


async def test_retrieve_ranks_pages_by_score() -> None:
    """Highest-scoring page should land at rank 1."""
    scores = {
        "p1::p1::page": 0.2,
        "p1::p7::page": 0.95,
        "p1::p3::page": 0.5,
    }
    retriever = _make_retriever(scores)

    out = await retriever.retrieve(Query(text="anything", top_k=3))

    import pytest

    assert [r.chunk_id for r in out] == ["p1::p7::page", "p1::p3::page", "p1::p1::page"]
    assert [r.score for r in out] == pytest.approx([0.95, 0.5, 0.2], abs=1e-6)


async def test_retrieve_caps_at_top_k() -> None:
    scores = {f"p1::p{i}::page": 1.0 - i * 0.01 for i in range(20)}
    retriever = _make_retriever(scores)
    out = await retriever.retrieve(Query(text="q", top_k=5))
    assert len(out) == 5


async def test_retrieve_returns_empty_when_no_pages() -> None:
    proc = _StubProcessor(chunk_ids=[], score_by_chunk={})
    retriever = VisualRetriever(
        model=_StubModel(), processor=proc, page_embeds={}, page_meta={}, device="cpu"
    )
    out = await retriever.retrieve(Query(text="q", top_k=10))
    assert out == []


async def test_retrieve_result_has_visual_source_metadata() -> None:
    scores = {"paperA::p4::page": 0.7}
    retriever = _make_retriever(scores)
    [hit] = await retriever.retrieve(Query(text="q", top_k=1))
    assert hit.source == "visual"
    assert hit.paper_id == "paperA"
    assert hit.page_numbers == [4]
    assert "Page image" in hit.text


async def test_score_query_uses_single_batched_call() -> None:
    """Performance regression guard: score_multi_vector must be called *once*
    per query, not once per page. Earlier per-page-loop was 5x slower."""
    scores = {f"p1::p{i}::page": float(i) for i in range(50)}
    retriever = _make_retriever(scores)

    call_count = 0
    real_score = retriever._processor.score_multi_vector

    def counting_score(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        result: torch.Tensor = real_score(*args, **kwargs)
        return result

    retriever._processor.score_multi_vector = counting_score
    await retriever.retrieve(Query(text="q", top_k=10))
    assert call_count == 1, f"score_multi_vector called {call_count} times; should be 1"
