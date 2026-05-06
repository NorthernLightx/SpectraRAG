"""Query expansion for multi-query retrieval.

Two strategies:

- **rewrite**: ask the LLM to produce N alternate phrasings of the user query,
  one per line. Each phrasing is independently retrieved against; results are
  fused via RRF. Helps when the original query uses different vocabulary than
  the source paper.
- **HyDE**: ask the LLM to write a hypothetical answer paragraph; embed *that*
  for dense retrieval. Closer in style to real paper passages than the question
  is, so semantic similarity finds better matches on multi-hop questions.

The two are independent; `MultiQueryRetriever` accepts either or both.
"""

from __future__ import annotations

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.prompts.loader import Prompt, load_prompt_by_name

_log = get_logger(__name__)


def _parse_rewrites(raw: str, *, expected: int) -> list[str]:
    """Pull rewritten queries off `raw` (one per line). Robust to numbered/bullet
    prefixes a model might add despite the prompt; dedupes case-insensitively."""
    seen: set[str] = set()
    rewrites: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        # Strip common leading garbage: "1.", "1)", "-", "*", quotes
        for prefix in ('"', "'", "- ", "* ", "• "):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip()
        # Strip leading "1.", "1)", "1:" and friends
        if text and text[0].isdigit():
            for sep in (". ", ") ", ": ", "- "):
                pos = text.find(sep)
                if 0 < pos <= 3:
                    text = text[pos + len(sep) :].lstrip()
                    break
        text = text.strip().strip('"').strip("'")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rewrites.append(text)
        if len(rewrites) >= expected:
            break
    return rewrites


class QueryExpander:
    """LLM-backed query expansion. Concrete impl, no protocol — there is only
    one (`QueryExpander`) and the rule of three says we abstract on the third.

    Both methods accept a `query` string and return either `list[str]`
    (`rewrite`) or `str` (`hyde`). On LLM failure, returns an empty list / empty
    string — caller decides whether to fall back to the original query.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        rewrite_prompt: Prompt | None = None,
        hyde_prompt: Prompt | None = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm
        self._model = model
        self._rewrite_prompt = rewrite_prompt or load_prompt_by_name("query_rewrite")
        self._hyde_prompt = hyde_prompt or load_prompt_by_name("query_hyde")
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def rewrite(self, query: str, *, n: int = 3) -> list[str]:
        """Return up to `n` alternate phrasings. Original query NOT included."""
        if n <= 0:
            return []
        system, user = self._rewrite_prompt.render(query=query, n=n)
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))
        with timed_event(_log, "query_rewrite.done", n_requested=n, model=self._model) as ctx:
            response = await self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            rewrites = _parse_rewrites(response.text, expected=n)
            ctx["n_returned"] = len(rewrites)
            ctx["tokens_in"] = response.tokens_in
            ctx["tokens_out"] = response.tokens_out
        return rewrites

    async def hyde(self, query: str) -> str:
        """Return a hypothetical answer paragraph; empty string on LLM failure."""
        system, user = self._hyde_prompt.render(query=query)
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))
        with timed_event(_log, "query_hyde.done", model=self._model) as ctx:
            response = await self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            text = (response.text or "").strip()
            ctx["chars"] = len(text)
            ctx["tokens_in"] = response.tokens_in
            ctx["tokens_out"] = response.tokens_out
        return text
