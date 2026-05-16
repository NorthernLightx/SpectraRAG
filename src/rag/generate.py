"""Generator: assemble context from retrieved chunks, call the LLM, parse citations."""

from __future__ import annotations

import re
import time
from pathlib import Path

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.observability.metrics import (
    GENERATE_LATENCY_MS,
    TOKENS_IN,
    TOKENS_OUT,
)
from src.prompts.loader import Prompt
from src.types import Answer, Citation, RetrievalResult

_log = get_logger(__name__)

# Match `[<id>]` and `[chunk_id <id>]` (some local models inline the literal "chunk_id"
# despite the prompt). The id can contain dots — ArXiv paper ids like `2604.22753v1`
# have them, and chunk ids are `<paper_id>::p<n>::c<n>`. Without `.` the regex would
# silently truncate `2604.22753v1::p5::c24` to `2604`.
_CITATION_RE = re.compile(r"\[(?:chunk_id\s+)?([A-Za-z0-9.:_\-]+)\]")
# Char budget proxy for a token budget: _max_context_chars =
# max_context_tokens * _CHARS_PER_TOKEN bounds how many retrieved chunks
# _build_context packs into the prompt. ~4 chars/token is the standard
# English heuristic; OpenRouter fronts many tokenizers so an exact per-model
# count would still only approximate here. This is a soft packing budget, not
# a hard model-context guard — the provider enforces the real token limit;
# this only decides how many chunks are worth sending.
_CHARS_PER_TOKEN = 4
# Visual chunk-id format (mirrors src/rag/retrievers/visual.py:_PAGE_CHUNK_FMT)
_PAGE_RE = re.compile(r"^(?P<paper>.+?)::p(?P<page>\d+)::page$")
_MAX_VISION_IMAGES = 4  # cap content blocks; 4 pages is plenty for a single answer


class Generator:
    """LLM-backed answer generator. Renders a prompt over retrieved chunks and parses citations.

    When `pages_dir` is set and any retrieved result has `source == "visual"`, the
    corresponding page PNG is attached to the LLM call as a content-block image so
    a vision-capable model (e.g. qwen3-vl, claude-sonnet-4.x vision) can read the
    image directly. Falls back to text-only when no visual results are present or
    `pages_dir` is None — preserves the existing behaviour.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompt: Prompt,
        model: str,
        temperature: float = 0.2,
        max_context_tokens: int = 8000,
        refusal_score_threshold: float | None = None,
        refusal_text: str = "I cannot answer this question from the provided corpus.",
        pages_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._model = model
        self._temperature = temperature
        self._max_context_chars = max_context_tokens * _CHARS_PER_TOKEN
        self._refusal_score_threshold = refusal_score_threshold
        self._refusal_text = refusal_text
        self._pages_dir = pages_dir

    async def answer(self, query: str, retrieved: list[RetrievalResult]) -> Answer:
        if self._refusal_score_threshold is not None and self._should_refuse(retrieved):
            _log.info(
                "generate.refused",
                reason="rerank_score_below_threshold",
                threshold=self._refusal_score_threshold,
                top_score=max((r.score for r in retrieved), default=None),
                n_retrieved=len(retrieved),
            )
            return self._refusal()
        context, used = self._build_context(retrieved)
        system, user = self._prompt.render(query=query, context=context)

        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))

        # When visual retrievals are present and pages_dir is configured, attach
        # the rendered page PNGs so a vision-capable LLM can read them directly.
        # Cap at _MAX_VISION_IMAGES to keep input tokens bounded.
        images = self._collect_image_paths(used)

        with timed_event(
            _log,
            "generate.done",
            model=self._model,
            prompt_version=self._prompt.version,
            context_chunks=len(used),
            images=len(images),
        ) as ctx:
            started = time.monotonic()
            response = await self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                images=images if images else None,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            citations = self._extract_citations(response.text, used)
            ctx["model"] = response.model
            ctx["tokens_in"] = response.tokens_in
            ctx["tokens_out"] = response.tokens_out
            ctx["citations"] = len(citations)
            attrs = {"model": response.model, "prompt_version": self._prompt.version}
            TOKENS_IN.add(response.tokens_in, attributes=attrs)
            TOKENS_OUT.add(response.tokens_out, attributes=attrs)
            GENERATE_LATENCY_MS.record(latency_ms, attributes=attrs)
        return Answer(
            text=response.text,
            citations=citations,
            model=response.model,
            prompt_version=self._prompt.version,
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def _collect_image_paths(self, used: list[RetrievalResult]) -> list[Path]:
        """For each visual RetrievalResult in `used`, return the rendered PNG path.

        Skips retrievals that aren't visual, missing pages_dir, malformed chunk_ids,
        and missing image files (logs a warning for the last). Capped at
        _MAX_VISION_IMAGES to keep input tokens bounded.
        """
        if self._pages_dir is None:
            return []
        out: list[Path] = []
        for r in used:
            if r.source != "visual":
                continue
            m = _PAGE_RE.match(r.chunk_id)
            if m is None:
                continue
            paper = m.group("paper")
            page = int(m.group("page"))
            img_path = self._pages_dir / paper / f"{paper}_p{page}.png"
            if not img_path.exists():
                _log.warning("generate.image_missing", path=str(img_path), chunk=r.chunk_id)
                continue
            out.append(img_path)
            if len(out) >= _MAX_VISION_IMAGES:
                _log.info("generate.images_capped", cap=_MAX_VISION_IMAGES, chunk=r.chunk_id)
                break
        return out

    def _build_context(self, retrieved: list[RetrievalResult]) -> tuple[str, list[RetrievalResult]]:
        used: list[RetrievalResult] = []
        parts: list[str] = []
        budget = 0
        for r in retrieved:
            pages = ", ".join(str(p) for p in r.page_numbers)
            block = f"[{r.chunk_id}] (pages {pages}) {r.text}"
            if budget + len(block) > self._max_context_chars and parts:
                break
            parts.append(block)
            used.append(r)
            budget += len(block) + 2
        return "\n\n".join(parts), used

    def _extract_citations(self, text: str, used: list[RetrievalResult]) -> list[Citation]:
        cited_ids = set(_CITATION_RE.findall(text))
        by_id = {r.chunk_id: r for r in used}
        citations: list[Citation] = []
        for cid in cited_ids:
            r = by_id.get(cid)
            if r is None:
                continue
            # ADR 0009: when the cited chunk is a region-grounded figure or
            # table, copy its bbox into the Citation so the demo UI can
            # render a region-precise highlight on the page image. `bbox`
            # is stored as a 4-list in chunk metadata by figure_to_chunk /
            # table_to_chunk; we validate the shape before passing through
            # to keep Citation's typed contract clean.
            bbox_raw = r.metadata.get("bbox")
            bbox: list[float] | None = None
            if (
                isinstance(bbox_raw, list)
                and len(bbox_raw) == 4
                and all(isinstance(v, (int, float)) for v in bbox_raw)
            ):
                bbox = [float(v) for v in bbox_raw]
            citations.append(
                Citation(
                    chunk_id=r.chunk_id,
                    paper_id=r.paper_id,
                    page_numbers=r.page_numbers,
                    bbox=bbox,
                )
            )
        return citations

    def _should_refuse(self, retrieved: list[RetrievalResult]) -> bool:
        if not retrieved:
            return True
        threshold = self._refusal_score_threshold
        assert threshold is not None  # narrowed by caller's check
        return all(r.score < threshold for r in retrieved)

    def _refusal(self) -> Answer:
        return Answer(
            text=self._refusal_text,
            citations=[],
            model="refusal-gate",
            prompt_version="refusal-v1",
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
        )
