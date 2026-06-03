"""Unit tests for the DCI corpus tools and the ReAct agent loop.

The agent is tested with a scripted fake LLM so the loop, action parsing, and
ranking logic are verified deterministically without Ollama.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.dci.agent import DciAgent
from src.dci.tools import CorpusTools
from src.llm.protocol import ChatResponse, Message
from src.rag.retrievers.dci import DciRetriever, build_dci_corpus
from src.types import Chunk, Query

_DOCS = {
    "paris": "The Eiffel Tower is in Paris.\nIt was completed in 1889.",
    "builder": "Gustave Eiffel was a French engineer.\nHis firm built the Eiffel Tower.",
    "noise": "The Great Wall of China is long.\nPandas live in China.",
}


class _ScriptedLLM:
    """LLMClient that returns canned responses in order (last repeats)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._i = 0

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        images: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        text = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return ChatResponse(text=text, model=model, tokens_in=1, tokens_out=1)


def test_search_ranks_by_match_count() -> None:
    tools = CorpusTools(_DOCS)
    hits = tools.search("Eiffel Tower Paris", top_k=3)
    assert hits[0].doc_id in {"paris", "builder"}
    assert all(h.score > 0 for h in hits)
    assert "noise" not in {h.doc_id for h in hits}


def test_grep_fixed_and_regex() -> None:
    tools = CorpusTools(_DOCS)
    fixed = tools.grep("Eiffel Tower", fixed=True)
    assert {h.doc_id for h in fixed} == {"paris", "builder"}
    regex = tools.grep(r"\b18\d\d\b")
    assert regex and regex[0].doc_id == "paris" and regex[0].line == 2


def test_read_bounds_and_missing() -> None:
    tools = CorpusTools(_DOCS)
    assert tools.read("paris", start=1, end=1).startswith("1: The Eiffel")
    assert "no such document" in tools.read("ghost")


def test_agent_retrieval_emits_padded_ranking() -> None:
    tools = CorpusTools(_DOCS)
    llm = _ScriptedLLM(
        [
            "THOUGHT: search\nACTION: SEARCH Eiffel Tower",
            "THOUGHT: rank\nACTION: RANK builder, paris",
        ]
    )
    agent = DciAgent(tools, llm, "fake", max_steps=5)
    res = asyncio.run(agent.run("Who built the Eiffel Tower?", mode="retrieval", top_k=3))
    assert res.stopped == "final"
    assert res.ranked_doc_ids[:2] == ["builder", "paris"]


def test_agent_qa_returns_answer() -> None:
    tools = CorpusTools(_DOCS)
    llm = _ScriptedLLM(
        [
            'THOUGHT: look\nACTION: GREP "built the Eiffel Tower"',
            "THOUGHT: done\nACTION: ANSWER Gustave Eiffel",
        ]
    )
    agent = DciAgent(tools, llm, "fake", max_steps=5)
    res = asyncio.run(agent.run("Who built it?", mode="qa"))
    assert res.answer == "Gustave Eiffel"
    assert res.stopped == "final"


def test_agent_recovers_from_unparsed_action() -> None:
    tools = CorpusTools(_DOCS)
    llm = _ScriptedLLM(
        [
            "I will just chat without an action.",  # unparsed -> correction
            "ACTION: ANSWER recovered",
        ]
    )
    agent = DciAgent(tools, llm, "fake", max_steps=5)
    res = asyncio.run(agent.run("q", mode="qa"))
    assert res.answer == "recovered"
    assert any(s.action == "(unparsed)" for s in res.steps)


def test_filter_all_requires_every_term() -> None:
    tools = CorpusTools(_DOCS)
    # both docs contain "Eiffel" AND "Tower"
    assert {h.doc_id for h in tools.filter_all(["Eiffel", "Tower"])} == {"paris", "builder"}
    # only the paris doc contains "Eiffel" AND "Paris" (builder never says Paris)
    assert {h.doc_id for h in tools.filter_all(["Eiffel", "Paris"])} == {"paris"}
    assert tools.filter_all(["Eiffel", "China"]) == []  # no doc has both


def test_count_selectivity() -> None:
    tools = CorpusTools(_DOCS)
    assert tools.count("Eiffel")[0] == 2  # in paris + builder
    assert tools.count("Pandas")[0] == 1
    assert tools.count("zzz")[0] == 0


def test_script_intersect_and_sandbox() -> None:
    tools = CorpusTools(_DOCS)
    out = tools.run_script("result = sorted(set(search('Eiffel')) & set(search('Paris')))")
    assert "paris" in out and "builder" not in out  # only paris has both terms
    assert "ImportError" in tools.run_script("import os\nresult = 1")  # imports blocked
    assert "did not assign" in tools.run_script("x = 1")  # must set result
    assert "timed out" in tools.run_script("while True:\n    pass", timeout=0.5)


def test_agent_uses_filter_then_ranks() -> None:
    tools = CorpusTools(_DOCS)
    llm = _ScriptedLLM(
        [
            "THOUGHT: narrow\nACTION: FILTER Eiffel, built",
            "THOUGHT: rank\nACTION: RANK builder",
        ]
    )
    agent = DciAgent(tools, llm, "fake", max_steps=4)
    res = asyncio.run(agent.run("who built it", mode="retrieval", top_k=2))
    assert res.ranked_doc_ids[0] == "builder"
    assert any(s.action == "FILTER" for s in res.steps)


def test_repeating_model_is_force_finalized() -> None:
    tools = CorpusTools(_DOCS)
    llm = _ScriptedLLM(["THOUGHT: loop\nACTION: SEARCH Eiffel"])  # same action forever
    agent = DciAgent(tools, llm, "fake", max_steps=8)
    res = asyncio.run(agent.run("q", mode="retrieval", top_k=3))
    assert res.stopped == "looped"  # loop-breaker fired instead of burning the budget
    assert res.ranked_doc_ids  # still ranks, padded from discovered docs


def test_dci_retriever_maps_agent_ranking_to_chunks() -> None:
    chunks = {
        "pa::p1::c1": Chunk(
            chunk_id="pa::p1::c1",
            paper_id="pa",
            page_numbers=[1],
            text="The Eiffel Tower is in Paris.",
        ),
        "pb::p2::c1": Chunk(
            chunk_id="pb::p2::c1",
            paper_id="pb",
            page_numbers=[2],
            text="Gustave Eiffel built the Eiffel Tower.",
        ),
        "pc::p3::c1": Chunk(
            chunk_id="pc::p3::c1", paper_id="pc", page_numbers=[3], text="Pandas live in China."
        ),
    }
    corpus, sur_to_chunk = build_dci_corpus(chunks)
    assert sur_to_chunk == {"c0": "pa::p1::c1", "c1": "pb::p2::c1", "c2": "pc::p3::c1"}
    llm = _ScriptedLLM(
        [
            "THOUGHT: search\nACTION: SEARCH Eiffel Tower",
            "THOUGHT: rank\nACTION: RANK c1, c0",
        ]
    )
    retr = DciRetriever(corpus, sur_to_chunk, chunks, llm, "fake", max_steps=4)
    results = asyncio.run(retr.retrieve(Query(text="who built the tower", top_k=2)))
    assert [r.chunk_id for r in results] == ["pb::p2::c1", "pa::p1::c1"]
    assert results[0].metadata["retriever"] == "dci"
    assert results[0].score > results[1].score  # rank-reciprocal preserves order


def test_dci_retriever_run_surfaces_step_trace() -> None:
    chunks = {
        "pa::p1::c1": Chunk(
            chunk_id="pa::p1::c1", paper_id="pa", page_numbers=[1], text="Eiffel Tower, Paris."
        ),
        "pb::p2::c1": Chunk(
            chunk_id="pb::p2::c1", paper_id="pb", page_numbers=[2], text="Gustave Eiffel built it."
        ),
    }
    corpus, sur_to_chunk = build_dci_corpus(chunks)
    llm = _ScriptedLLM(
        [
            "THOUGHT: search\nACTION: SEARCH Eiffel",
            "THOUGHT: rank\nACTION: RANK c1, c0",
        ]
    )
    retr = DciRetriever(corpus, sur_to_chunk, chunks, llm, "fake", max_steps=4)
    results, dci_result = asyncio.run(retr.run(Query(text="who built it", top_k=2)))
    assert [r.chunk_id for r in results] == ["pb::p2::c1", "pa::p1::c1"]
    assert [s.action for s in dci_result.steps] == ["SEARCH", "RANK"]
    assert dci_result.steps[0].observation  # the SEARCH observation is populated


def test_budget_exhaustion_still_ranks() -> None:
    tools = CorpusTools(_DOCS)
    # two alternating actions never finalize but never trip the repeat detector
    llm = _ScriptedLLM(["ACTION: SEARCH Eiffel", "ACTION: SEARCH Tower"])
    agent = DciAgent(tools, llm, "fake", max_steps=2)
    res = asyncio.run(agent.run("q", mode="retrieval", top_k=3))
    assert res.stopped == "budget"
    assert res.ranked_doc_ids  # padded from discovered docs
