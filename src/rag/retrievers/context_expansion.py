"""Context-neighbourhood expansion: a retrieved chunk rarely answers alone.

A human reading a paper doesn't fixate on one sentence — they read around
it (the prior paragraph that sets up a definition, the next one that states
the result) and they look at the figure/table the text points to. The base
retrievers return single best chunks in isolation and the generator
concatenates exactly those (`generate.py:_build_context` does no expansion).
This decorator adds the *neighbourhood* before generation:

  - **window**: the ±k sequential chunks on the same page of the same paper
    (chunk ids are page-local sequential — `<paper>::p<page>::c<idx>`), so
    a matched sentence arrives with the prose around it.
  - **linked artifacts**: if an anchor's text says "Figure 3" / "Table 7",
    pull that figure/table chunk (whose text starts "Figure 3: …" per
    `figure_to_chunk`), so a paragraph that *references* a visual brings
    the visual's caption/VLM-text with it.

Pure decorator over the `Retriever` Protocol (mirrors RegionNumberBoost /
MultiQuery). Anchors keep their order and score; expansions are appended
with a decayed score so downstream stays anchor-first. Total is capped so
the generator's char budget isn't blown. With `window=0` and
`link_artifacts=False` it is an exact passthrough — the eval's baseline arm.
"""

from __future__ import annotations

import re

from src.observability.logging import get_logger
from src.rag.retrievers.protocol import Retriever
from src.types import Chunk, Query, RetrievalResult

_log = get_logger(__name__)

_ID_RE = re.compile(r"^(?P<paper>.+)::p(?P<page>\d+)::c(?P<idx>\d+)$")
_TABLE_RE = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\bfig(?:ure|\.)?\s*(\d+)\b", re.IGNORECASE)
_DECAY = 0.5  # expansion score = anchor score * decay → stays below anchors


def _label_head(text: str, kind: str, number: int) -> bool:
    head = text.lstrip()
    return (
        bool(head)
        and re.match(rf"^{kind}\.?\s+{number}\s*[:.—\-]", head, re.IGNORECASE) is not None
    )


class ContextExpansionRetriever:
    """Wraps a Retriever; appends each anchor's page-neighbours and the
    figures/tables its text references. ADR 0016.
    """

    def __init__(
        self,
        *,
        base: Retriever,
        chunks_by_id: dict[str, Chunk],
        window: int = 1,
        link_artifacts: bool = True,
        max_expansions: int = 12,
    ) -> None:
        self._base = base
        self._chunks = chunks_by_id
        self._window = window
        self._link = link_artifacts
        self._cap = max_expansions
        # paper_id -> [(chunk_id, text)] for artifact lookup, built once
        self._by_paper: dict[str, list[tuple[str, str]]] = {}
        for cid, ch in chunks_by_id.items():
            self._by_paper.setdefault(ch.paper_id, []).append((cid, ch.text))

    def _neighbours(self, chunk_id: str) -> list[str]:
        m = _ID_RE.match(chunk_id)
        if not m or self._window <= 0:
            return []
        paper, page, idx = m["paper"], int(m["page"]), int(m["idx"])
        out: list[str] = []
        for d in range(1, self._window + 1):
            for j in (idx - d, idx + d):
                cid = f"{paper}::p{page}::c{j}"
                if j >= 0 and cid in self._chunks:
                    out.append(cid)
        return out

    def _linked(self, anchor_text: str, paper_id: str) -> list[str]:
        if not self._link:
            return []
        wanted: list[tuple[str, int]] = [
            ("table", int(n)) for n in _TABLE_RE.findall(anchor_text)
        ] + [("figure", int(n)) for n in _FIGURE_RE.findall(anchor_text)]
        if not wanted:
            return []
        out: list[str] = []
        for cid, text in self._by_paper.get(paper_id, []):
            for kind, num in wanted:
                if _label_head(text, kind, num):
                    out.append(cid)
                    break
        return out

    def _as_result(self, chunk_id: str, score: float) -> RetrievalResult | None:
        ch = self._chunks.get(chunk_id)
        if ch is None:
            return None
        meta: dict[str, object] = dict(ch.metadata)
        if ch.section:
            meta["section"] = ch.section
        return RetrievalResult(
            chunk_id=ch.chunk_id,
            paper_id=ch.paper_id,
            score=score,
            text=ch.text,
            page_numbers=ch.page_numbers,
            source="pipeline",
            metadata=meta,
        )

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        anchors = await self._base.retrieve(query)
        if self._window <= 0 and not self._link:
            return anchors  # exact passthrough — the eval baseline arm

        seen = {r.chunk_id for r in anchors}
        expansions: list[RetrievalResult] = []
        for a in anchors:
            if len(expansions) >= self._cap:
                break
            extra = self._neighbours(a.chunk_id) + self._linked(a.text, a.paper_id)
            for cid in extra:
                if cid in seen or len(expansions) >= self._cap:
                    continue
                r = self._as_result(cid, a.score * _DECAY)
                if r is not None:
                    seen.add(cid)
                    expansions.append(r)
        if expansions:
            _log.info(
                "context_expansion.applied",
                anchors=len(anchors),
                added=len(expansions),
                window=self._window,
                link=self._link,
            )
        return anchors + expansions
