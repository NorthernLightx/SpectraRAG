"""Generator: build the reader context, call the LLM, parse citations."""

from __future__ import annotations

import re
import string
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from src.llm.protocol import ContentPart, ImagePart, LLMClient, Message, TextPart
from src.observability.logging import get_logger, timed_event
from src.observability.metrics import (
    GENERATE_LATENCY_MS,
    TOKENS_IN,
    TOKENS_OUT,
)
from src.prompts.loader import Prompt
from src.rag.context import (
    MAX_PAGE_IMAGES,
    FigureRef,
    PageImageRef,
    PriorTurn,
    ReaderContext,
    ReaderMessage,
    build_reader_context,
)
from src.types import Answer, Citation, RetrievalResult

_log = get_logger(__name__)

# Match `[<id>]` and `[chunk_id <id>]` (some local models inline the literal "chunk_id"
# despite the prompt). The id can contain dots: ArXiv paper ids like `2604.22753v1`
# have them, and chunk ids are `<paper_id>::p<n>::c<n>`. Without `.` the regex would
# silently truncate `2604.22753v1::p5::c24` to `2604`.
_CITATION_RE = re.compile(r"\[(?:chunk_id\s+)?([A-Za-z0-9.:_\-]+)\]")
_PAGE_ID_RE = re.compile(r"^(?P<paper>.+?)::p(?P<page>\d+)::page$")
# Benchmark corpora ship pre-rendered JPEGs (scripts.fetch_mmdocir) while
# render_pages writes PNG, so both are accepted.
_PAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class Generator:
    """LLM-backed answer generator over `src.rag.context.build_reader_context`
    (ADR 0033), the same context the chat UI sends.

    With `pages_dir` set, the page image of every retrieved result goes inline
    after its chunk, up to `max_vision_images`, each behind a citable label.
    `figure_index` supplies the figure and table chunks a question can name by
    number; their captions and pages join the context.
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
        max_vision_images: int = MAX_PAGE_IMAGES,
        figure_index: Callable[[], Sequence[FigureRef]] | None = None,
    ) -> None:
        fields = {f for _, f, _, _ in string.Formatter().parse(prompt.user_template) if f}
        if fields != {"query"}:
            raise ValueError(
                f"Prompt {prompt.name!r} fills {sorted(fields)}; a reader prompt's "
                "user_template takes only {query}, the context builder places the "
                "chunks and images (ADR 0033)."
            )
        self._llm = llm
        self._prompt = prompt
        self._model = model
        self._temperature = temperature
        self._max_context_tokens = max_context_tokens
        self._refusal_score_threshold = refusal_score_threshold
        self._refusal_text = refusal_text
        self._pages_dir = pages_dir
        # ADR 0024's route-by-fit raises this to the page budget so a
        # whole-document feed isn't silently truncated.
        self._max_vision_images = max_vision_images
        self._figure_index = figure_index

    def context(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        prior_turns: Sequence[PriorTurn] = (),
    ) -> ReaderContext:
        return build_reader_context(
            query,
            retrieved,
            prompt=self._prompt,
            prior_turns=prior_turns,
            figures=self._figure_index() if self._figure_index is not None else (),
            include_images=self._pages_dir is not None,
            max_images=self._max_vision_images,
            max_context_tokens=self._max_context_tokens,
            image_available=lambda paper, page: self._page_path(paper, page) is not None,
        )

    async def answer(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        *,
        prior_turns: Sequence[PriorTurn] = (),
    ) -> Answer:
        if self._refusal_score_threshold is not None and self._should_refuse(retrieved):
            _log.info(
                "generate.refused",
                reason="rerank_score_below_threshold",
                threshold=self._refusal_score_threshold,
                top_score=max((r.score for r in retrieved), default=None),
                n_retrieved=len(retrieved),
            )
            return self._refusal()
        ctx = self.context(query, retrieved, prior_turns=prior_turns)
        messages = [self._resolve(m) for m in ctx.messages]
        n_images = sum(
            isinstance(part, ImagePart)
            for m in messages
            if isinstance(m.content, list)
            for part in m.content
        )

        with timed_event(
            _log,
            "generate.done",
            model=self._model,
            prompt_version=self._prompt.version,
            context_chunks=len(ctx.used),
            images=n_images,
            injected=len(ctx.injected),
        ) as log_ctx:
            started = time.monotonic()
            response = await self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            citations = self._extract_citations(response.text, ctx)
            log_ctx["model"] = response.model
            log_ctx["tokens_in"] = response.tokens_in
            log_ctx["tokens_out"] = response.tokens_out
            log_ctx["citations"] = len(citations)
            attrs = {"model": response.model, "prompt_version": self._prompt.version}
            TOKENS_IN.add(response.tokens_in, attributes=attrs)
            TOKENS_OUT.add(response.tokens_out, attributes=attrs)
            GENERATE_LATENCY_MS.record(latency_ms, attributes=attrs)
        return Answer(
            text=response.text,
            citations=citations,
            context_ids=ctx.context_ids,
            model=response.model,
            prompt_version=self._prompt.version,
            latency_ms=latency_ms,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def _page_path(self, paper: str, page: int) -> Path | None:
        if self._pages_dir is None:
            return None
        for suffix in _PAGE_SUFFIXES:
            path = self._pages_dir / paper / f"{paper}_p{page}{suffix}"
            if path.exists():
                return path
        _log.warning("generate.image_missing", paper=paper, page=page)
        return None

    def _resolve(self, message: ReaderMessage) -> Message:
        """Page image refs become the label text plus the image file."""
        if isinstance(message.content, str):
            return Message(role=message.role, content=message.content)
        parts: list[ContentPart] = []
        for part in message.content:
            if isinstance(part, PageImageRef):
                path = self._page_path(part.paper_id, part.page)
                if path is not None:
                    parts.append(TextPart(text=part.label))
                    parts.append(ImagePart(path=path))
            else:
                parts.append(part)
        return Message(role=message.role, content=parts)

    def _extract_citations(self, text: str, ctx: ReaderContext) -> list[Citation]:
        """Citations for the bracketed ids the reader was actually shown:
        retrieved chunks, injected figure captions, attached page images."""
        cited = set(_CITATION_RE.findall(text)) & set(ctx.context_ids)
        by_id = {r.chunk_id: r for r in ctx.used}
        figures = {f.chunk_id: f for f in ctx.injected}
        citations: list[Citation] = []
        for cid in cited:
            r = by_id.get(cid)
            if r is not None:
                citations.append(
                    Citation(
                        chunk_id=r.chunk_id,
                        paper_id=r.paper_id,
                        page_numbers=r.page_numbers,
                        bbox=_bbox(r.metadata.get("bbox")),
                    )
                )
                continue
            fig = figures.get(cid)
            if fig is not None:
                citations.append(
                    Citation(
                        chunk_id=fig.chunk_id,
                        paper_id=fig.paper_id,
                        page_numbers=[fig.page],
                        bbox=fig.bbox,
                    )
                )
                continue
            page = _PAGE_ID_RE.match(cid)
            if page is not None:
                citations.append(
                    Citation(
                        chunk_id=cid,
                        paper_id=page.group("paper"),
                        page_numbers=[int(page.group("page"))],
                    )
                )
        return citations

    def _should_refuse(self, retrieved: list[RetrievalResult]) -> bool:
        """Refuse when nothing was retrieved, or when every result is a
        reranked chunk scoring under the threshold. The threshold is calibrated
        on the cross-encoder's scale only (ADR 0006, 2026-09-23 amendment): a
        visual page or an unreranked chunk has no calibrated score, so one in
        the list keeps the gate out of the way rather than letting weak text
        overrule strong visual evidence."""
        if not retrieved:
            return True
        threshold = self._refusal_score_threshold
        assert threshold is not None  # narrowed by caller's check
        if any(r.score_kind != "rerank" for r in retrieved):
            return False
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


def _bbox(raw: object) -> list[float] | None:
    """ADR 0009: a region-grounded figure or table chunk carries its bbox as a
    4-list in metadata; the citation copies it so the UI can highlight the
    region. Anything else is no bbox."""
    if isinstance(raw, list) and len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
        return [float(v) for v in raw]
    return None
