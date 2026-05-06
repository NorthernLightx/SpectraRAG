"""LLM-based query classifier — replaces ADR 0008's regex when MMLongBench-style
queries don't carry their own modality cue.

The MMLongBench eval (run cc45831697b6) found that the regex classifier dispatched
only 26 of 149 queries to hybrid where 98 were figure/table-evidenced — a ~75 %
miss rate caused by natural-language queries that don't say "Figure X" or
"Table N" explicitly. This classifier reads the query through a small LLM
(default `openai/gpt-4o-mini`, ~$0.0001 per call, ~0.5-1.5 s) and emits a
Category. Falls back to "definitional" on any parse failure so the routing
decision degrades to text-only (the safe baseline).

Used optionally by `RoutingRetriever` — pass an instance via the `classifier`
constructor arg to override the default regex.
"""

from __future__ import annotations

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger
from src.prompts.loader import Prompt
from src.rag.retrievers.routing import Category

_log = get_logger(__name__)

_VALID_CATEGORIES: frozenset[str] = frozenset(
    {"table", "figure", "multi_hop", "factual", "definitional"}
)


class LLMQueryClassifier:
    """Async query classifier backed by an LLM."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        prompt: Prompt,
        temperature: float = 0.0,
        max_tokens: int = 16,
    ) -> None:
        self._llm = llm
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def classify(self, query: str) -> Category:
        """Map a query text to one of the five Categories. Defaults to
        'definitional' on any parse error or empty response so misclassification
        degrades safely (text-only routing — the strong baseline)."""
        system, user = self._prompt.render(query=query)
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))

        response = await self._llm.chat(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return _parse_category(response.text)


def _parse_category(raw: str) -> Category:
    """Pull the first valid category token from the LLM output. Returns
    'definitional' as the safe fallback on any parse failure (empty response,
    unknown token, model rambled instead of emitting a single token)."""
    if not raw:
        return "definitional"
    # First non-empty stripped line, lowercased
    first_line = next((line.strip().lower() for line in raw.splitlines() if line.strip()), "")
    # Tolerate trailing punctuation / quotes / backticks
    cleaned = first_line.strip(".,;:!? \"'`")
    if cleaned in _VALID_CATEGORIES:
        # The Literal narrows on membership; mypy doesn't follow that, so cast.
        return cleaned  # type: ignore[return-value]
    _log.warning("classifier_llm.parse_fail", raw=raw[:120], cleaned=cleaned)
    return "definitional"
