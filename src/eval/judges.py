"""LLM-as-judge metrics: faithfulness, answer relevance, context precision.

Each judge calls the LLM once with a versioned prompt, parses a single decimal
score from the first line of the response, and returns it alongside the
rationale text. The first-line-decimal format is deliberately simple so small
local models (e.g. qwen2.5:7b) can produce parseable output reliably; strict
JSON tends to fail at this size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.prompts.loader import Prompt
from src.types import RetrievalResult

_log = get_logger(__name__)

_FIRST_DECIMAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class JudgeOutput:
    """One judgment: a clamped 0-1 score and the model's rationale."""

    score: float
    rationale: str
    model: str
    prompt_version: str


def _parse_score(raw: str) -> tuple[float, str]:
    """Pull the first decimal off the first non-empty line; clamp to [0, 1].

    Returns (score, rationale). If parsing fails, score=0.0 and rationale is
    the full raw response (so callers can debug).
    """
    if not raw or not raw.strip():
        return 0.0, "[judge returned empty response]"

    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return 0.0, "[judge returned only whitespace]"

    match = _FIRST_DECIMAL_RE.search(lines[0])
    if match is None:
        return 0.0, raw.strip()

    try:
        score = float(match.group(0))
    except ValueError:
        return 0.0, raw.strip()

    score = max(0.0, min(1.0, score))
    rationale = "\n".join(lines[1:]).strip() or "[no rationale]"
    return score, rationale


class LLMJudge:
    """LLM-as-judge for generation metrics. One LLM call per (query, metric)."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        faithfulness_prompt: Prompt,
        answer_relevance_prompt: Prompt,
        context_precision_prompt: Prompt,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm
        self._model = model
        self._faithfulness_prompt = faithfulness_prompt
        self._answer_relevance_prompt = answer_relevance_prompt
        self._context_precision_prompt = context_precision_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def faithfulness(
        self, *, query: str, answer: str, retrieved: list[RetrievalResult]
    ) -> JudgeOutput:
        context = _format_chunks(retrieved)
        return await self._judge(
            self._faithfulness_prompt,
            metric="faithfulness",
            query=query,
            context=context,
            answer=answer,
        )

    async def answer_relevance(self, *, query: str, answer: str) -> JudgeOutput:
        return await self._judge(
            self._answer_relevance_prompt,
            metric="answer_relevance",
            query=query,
            answer=answer,
        )

    async def context_precision(
        self, *, query: str, retrieved: list[RetrievalResult]
    ) -> JudgeOutput:
        context = _format_chunks(retrieved)
        return await self._judge(
            self._context_precision_prompt,
            metric="context_precision",
            query=query,
            context=context,
        )

    async def _judge(self, prompt: Prompt, *, metric: str, **render_kwargs: object) -> JudgeOutput:
        # Some prompts don't use 'answer' or 'context'; pass empty strings so str.format won't KeyError.
        defaults: dict[str, object] = {"query": "", "context": "", "answer": ""}
        defaults.update(render_kwargs)
        system, user = prompt.render(**defaults)

        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))

        with timed_event(
            _log,
            "judge.done",
            metric=metric,
            model=self._model,
            prompt_version=prompt.version,
        ) as ctx:
            response = await self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            score, rationale = _parse_score(response.text)
            ctx["score"] = score
            ctx["tokens_in"] = response.tokens_in
            ctx["tokens_out"] = response.tokens_out

        return JudgeOutput(
            score=score,
            rationale=rationale,
            model=response.model,
            prompt_version=prompt.version,
        )


def _format_chunks(retrieved: list[RetrievalResult]) -> str:
    """Render retrieved chunks as `[chunk_id] text` blocks for the judge prompts."""
    return "\n\n".join(f"[{r.chunk_id}] {r.text}" for r in retrieved)
