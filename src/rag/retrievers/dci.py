"""DciRetriever: experimental agentic grep retrieval over the corpus (opt-in).

Adapts the DCI agent (`src/dci`) to PrismRAG's retrieval interface. It materialises
the in-memory chunk index into a grep-able text corpus and lets an LLM agent search
it with SEARCH/GREP/READ, returning the chunks it ranks.

Deliberately NOT the default and NOT wired into routing. It is:
- **text-only** — blind to the visual leg's pixel content (a third of the answers);
- **slow** — a multi-step LLM loop vs sub-second vector retrieval;
- **LLM-bound** — the agent runs server-side, so the route must supply an OpenRouter
  key (the server's own, or the user's for this request).

Measured on BRIGHT it reaches the agentic-text-IR tier; on this multimodal corpus
retrieval is not the bottleneck (see the DCI ADR), so this is a showcase mode.
"""

from __future__ import annotations

from src.dci.agent import DciAgent
from src.dci.tools import CorpusTools
from src.llm.protocol import LLMClient
from src.types import Chunk, Query, RetrievalResult


def build_dci_corpus(chunks: dict[str, Chunk]) -> tuple[CorpusTools, dict[str, str]]:
    """Grep-able corpus from the chunk index. Short surrogate ids (`c0`, `c1`, …)
    keep the agent's RANK output terse and exact; the returned map translates them
    back to real chunk ids. Built once and cached by the caller (corpus is static)."""
    docs: dict[str, str] = {}
    sur_to_chunk: dict[str, str] = {}
    for i, (chunk_id, chunk) in enumerate(chunks.items()):
        sur = f"c{i}"
        docs[sur] = chunk.text
        sur_to_chunk[sur] = chunk_id
    return CorpusTools(docs), sur_to_chunk


class DciRetriever:
    """Runs the DCI agent over the corpus and maps its ranking to RetrievalResults."""

    def __init__(
        self,
        corpus: CorpusTools,
        sur_to_chunk: dict[str, str],
        chunks: dict[str, Chunk],
        llm: LLMClient,
        model: str,
        *,
        max_steps: int = 16,
    ) -> None:
        self._sur_to_chunk = sur_to_chunk
        self._chunks = chunks
        # read+grep is the cost-optimal config (the full-bash A/B was a wash); see ADR.
        self._agent = DciAgent(corpus, llm, model, max_steps=max_steps, toolset="readgrep")

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        result = await self._agent.run(query.text, mode="retrieval", top_k=query.top_k)
        out: list[RetrievalResult] = []
        for rank, sur in enumerate(result.ranked_doc_ids[: query.top_k]):
            chunk = self._chunks.get(self._sur_to_chunk.get(sur, ""))
            if chunk is None or not chunk.page_numbers:
                continue
            out.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=1.0 / (rank + 1),  # the agent ranks; rank-reciprocal as the score
                    text=chunk.text,
                    page_numbers=chunk.page_numbers,
                    source="pipeline",
                    metadata={**chunk.metadata, "retriever": "dci"},
                )
            )
        return out
