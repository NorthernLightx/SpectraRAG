"""A retriever that serves recorded results, for legs a CI runner cannot run.

The visual leg needs ColQwen2 (about 9 GB in fp32) and a page index that is not
in git. `scripts/record_visual_legs.py` records its results for a golden set;
`ReplayRetriever` serves them back so the fusion and routing code run live
against a fixed visual leg (`scripts/eval_retrieval_ci.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.types import Query, RetrievalResult


def _key(text: str, paper_filter: str | None) -> str:
    return f"{paper_filter or ''}\x1f{text}"


class ReplayRetriever:
    """Returns the results recorded for (query text, paper filter), cut to top_k.
    A query with no recording raises KeyError: the fixture is stale."""

    def __init__(self, recorded: dict[str, list[RetrievalResult]], depth: int) -> None:
        self._recorded = recorded
        self.depth = depth

    @classmethod
    def from_fixture(cls, path: Path) -> ReplayRetriever:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        recorded = {
            _key(q["text"], q.get("paper_filter")): [
                RetrievalResult.model_validate(r) for r in q["results"]
            ]
            for q in data["queries"]
        }
        return cls(recorded, depth=int(data["depth"]))

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        key = _key(query.text, query.paper_id_filter())
        if key not in self._recorded:
            raise KeyError(f"no recorded results for {query.text!r}; re-record the fixture")
        if query.top_k > self.depth:
            raise ValueError(f"asked for {query.top_k} results, fixture holds {self.depth}")
        return self._recorded[key][: query.top_k]


def fixture_entry(
    text: str, paper_filter: str | None, results: list[RetrievalResult]
) -> dict[str, Any]:
    return {
        "text": text,
        "paper_filter": paper_filter,
        "results": [r.model_dump(mode="json") for r in results],
    }
