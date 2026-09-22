"""ReplayRetriever serves a recorded leg for the CPU CI gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.replay import ReplayRetriever, fixture_entry
from src.types import Query, RetrievalResult


def _page(paper: str, page: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper}::p{page}::page",
        paper_id=paper,
        score=20.0 - page,
        text="",
        page_numbers=[page],
        source="visual",
        score_kind="maxsim",
    )


def _fixture(tmp_path: Path) -> Path:
    path = tmp_path / "legs.json"
    path.write_text(
        json.dumps(
            {
                "depth": 3,
                "queries": [
                    fixture_entry("what?", "a", [_page("a", p) for p in (1, 2, 3)]),
                    fixture_entry("what?", None, [_page("b", 9)]),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


async def test_replays_by_text_and_paper_filter(tmp_path: Path) -> None:
    replay = ReplayRetriever.from_fixture(_fixture(tmp_path))
    scoped = await replay.retrieve(Query(text="what?", top_k=2, filters={"paper_id": "a"}))
    assert [r.chunk_id for r in scoped] == ["a::p1::page", "a::p2::page"]
    assert scoped[0].score_kind == "maxsim"
    unscoped = await replay.retrieve(Query(text="what?", top_k=2))
    assert [r.chunk_id for r in unscoped] == ["b::p9::page"]


async def test_unrecorded_query_is_an_error(tmp_path: Path) -> None:
    replay = ReplayRetriever.from_fixture(_fixture(tmp_path))
    with pytest.raises(KeyError, match="re-record"):
        await replay.retrieve(Query(text="something new", top_k=2))


async def test_asking_deeper_than_recorded_is_an_error(tmp_path: Path) -> None:
    replay = ReplayRetriever.from_fixture(_fixture(tmp_path))
    with pytest.raises(ValueError, match="holds 3"):
        await replay.retrieve(Query(text="what?", top_k=5, filters={"paper_id": "a"}))
