"""LLM-as-judge metrics: faithfulness, answer relevance, context precision.

Each judge calls the LLM with a versioned prompt and parses a single decimal
score from the first line of the response. The first-line-decimal format is
deliberately simple so small local models (e.g. qwen2.5:7b) can produce
parseable output reliably; strict JSON tends to fail at this size.

Multi-seed averaging (B2 / Tier 2): when `n_samples > 1`, each metric is
sampled `n_samples` times in parallel at a non-zero `sampling_temperature`.
The output's `.score` is the mean across samples; `.score_std` is the
sample standard deviation. This converts the "did q33 happen to score 1.0
or 0.5 this run?" judge variance from a hidden source of metric noise into
a measurable quantity, so future feature deltas are interpretable.
"""

from __future__ import annotations

import asyncio
import math
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
    """One judgment: a clamped 0-1 score and the model's rationale.

    `score_std` is the sample standard deviation across `n_samples` calls
    when multi-seed averaging is enabled (B2). 0.0 when `n_samples == 1`
    (single call, no variance to measure). The mean is in `score`.
    `n_samples` records how many calls were averaged so downstream
    reporting can format `score ± score_std (n=N)` honestly.
    """

    score: float
    rationale: str
    model: str
    prompt_version: str
    score_std: float = 0.0
    n_samples: int = 1


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
    """LLM-as-judge for generation metrics.

    By default makes one LLM call per (query, metric). Set `n_samples > 1`
    to enable multi-seed averaging (B2 / Tier 2): each metric becomes N
    parallel calls at `sampling_temperature` (defaults to 0.7 — high enough
    to surface judge variance without making the scores nonsense), with
    the mean reported as `JudgeOutput.score` and the sample stddev as
    `JudgeOutput.score_std`.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        faithfulness_prompt: Prompt,
        answer_relevance_prompt: Prompt,
        context_precision_prompt: Prompt,
        answer_correctness_prompt: Prompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        n_samples: int = 1,
        sampling_temperature: float = 0.7,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self._llm = llm
        self._model = model
        self._faithfulness_prompt = faithfulness_prompt
        self._answer_relevance_prompt = answer_relevance_prompt
        self._context_precision_prompt = context_precision_prompt
        # Optional so existing callers (older eval runs) keep working; the
        # runner only invokes answer_correctness when the prompt is set.
        self._answer_correctness_prompt = answer_correctness_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._n_samples = n_samples
        # Single-seed callers stay at deterministic temperature=0; only
        # bump to sampling_temperature when multi-seed is requested.
        self._sampling_temperature = sampling_temperature if n_samples > 1 else temperature

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

    @property
    def has_answer_correctness(self) -> bool:
        """True iff this judge was configured with the answer_correctness prompt."""
        return self._answer_correctness_prompt is not None

    async def answer_correctness(
        self, *, query: str, answer: str, expected_facts: list[str]
    ) -> JudgeOutput:
        """Recall of ground-truth facts in the answer. Chunk-id-robust (ADR 0019).

        Caller must check `has_answer_correctness` (or only invoke when
        `expected_facts` is non-empty and the answer is not a refusal —
        the runner enforces both). Renders `expected_facts` as a bullet
        list so the judge prompt can count coverage.
        """
        if self._answer_correctness_prompt is None:
            raise RuntimeError("answer_correctness judge not configured")
        rendered = "\n".join(f"- {f}" for f in expected_facts)
        return await self._judge(
            self._answer_correctness_prompt,
            metric="answer_correctness",
            query=query,
            answer=answer,
            expected_facts=rendered,
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
            n_samples=self._n_samples,
        ) as ctx:
            # Multi-seed averaging (B2): when n_samples > 1, fire all N calls
            # in parallel at a non-zero sampling_temperature, then average.
            # When n_samples == 1, this collapses to a single call and matches
            # the previous behavior exactly.
            tasks = [
                self._llm.chat(
                    messages=messages,
                    model=self._model,
                    temperature=self._sampling_temperature,
                    max_tokens=self._max_tokens,
                )
                for _ in range(self._n_samples)
            ]
            responses = await asyncio.gather(*tasks)
            scored = [_parse_score(r.text) for r in responses]
            scores = [s for s, _ in scored]
            mean_score = sum(scores) / len(scores)
            score_std = _stddev(scores)
            # Report rationale + token usage from the first sample; aggregate
            # accounting (sum across samples) on the otel span so cost is
            # observable.
            _, first_rationale = scored[0]
            tokens_in_total = sum(r.tokens_in for r in responses)
            tokens_out_total = sum(r.tokens_out for r in responses)
            ctx["score"] = mean_score
            ctx["score_std"] = score_std
            ctx["tokens_in"] = tokens_in_total
            ctx["tokens_out"] = tokens_out_total

        return JudgeOutput(
            score=mean_score,
            score_std=score_std,
            n_samples=self._n_samples,
            rationale=first_rationale,
            model=responses[0].model,
            prompt_version=prompt.version,
        )


def _stddev(values: list[float]) -> float:
    """Sample standard deviation. Returns 0.0 for n < 2 — there's no
    variance to measure with a single sample, and ddof=1 would divide by 0."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _format_chunks(retrieved: list[RetrievalResult]) -> str:
    """Render retrieved chunks as `[chunk_id] text` blocks for the judge prompts."""
    return "\n\n".join(f"[{r.chunk_id}] {r.text}" for r in retrieved)
