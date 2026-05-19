"""AgenticRetriever: sub-query decomposition + RRF fusion + graceful fallback (ADR 0019)."""

from __future__ import annotations

from src.llm.protocol import ChatResponse, Message
from src.rag.retrievers.agentic import AgenticRetriever, _parse_subqueries
from src.types import Query, RetrievalResult


class _StubLLM:
    def __init__(self, replies: list[str], *, boom: bool = False) -> None:
        self._replies = list(replies)
        self._boom = boom
        self.calls: list[str] = []

    async def chat(self, messages: list[Message], model: str, **kwargs: object) -> ChatResponse:
        self.calls.append(messages[-1].content if messages else "")
        if self._boom:
            raise RuntimeError("llm down")
        text = self._replies.pop(0) if self._replies else ""
        return ChatResponse(text=text, model=model, tokens_in=1, tokens_out=1)


class _FakeBase:
    def __init__(self, plan: dict[str, list[tuple[str, float]]]) -> None:
        self._plan = plan
        self.calls: list[str] = []

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.calls.append(query.text)
        return [
            RetrievalResult(
                chunk_id=cid,
                paper_id="p1",
                score=score,
                text=f"text for {cid}",
                page_numbers=[1],
                source="pipeline",
            )
            for cid, score in self._plan.get(query.text, [])
        ]


def test_parse_subqueries_strips_prefixes_dedupes() -> None:
    out = _parse_subqueries(
        "1. What is X?\n- What is X?\n  * how does Y work?\n3) Same as line 1\n",
        original="orig",
        max_subqueries=4,
    )
    # Dedupe: "What is X?" survives only once; "Same as line 1" is its own item.
    assert out == ["What is X?", "how does Y work?", "Same as line 1"]


def test_parse_subqueries_empty_falls_back_to_original() -> None:
    assert _parse_subqueries("   \n\n", original="orig", max_subqueries=4) == ["orig"]


async def test_atomic_decomposition_bypasses_fan_out() -> None:
    base = _FakeBase({"q": [("c1", 0.9)]})
    llm = _StubLLM(["q"])  # decomposition returns the same single query
    agent = AgenticRetriever(base=base, llm=llm, model="m")
    out = await agent.retrieve(Query(text="q", top_k=5))
    assert [r.chunk_id for r in out] == ["c1"]
    # Atomic: only the original goes to base; no fan-out.
    assert base.calls == ["q"]


async def test_decomposition_fans_out_and_rrf_promotes_shared_hits() -> None:
    plan = {
        "q-original-ignored": [],
        "sub A": [("c1", 0.9), ("c2", 0.5)],
        "sub B": [("c2", 0.7), ("c3", 0.4)],
        "sub C": [("c2", 0.6), ("c4", 0.3)],
    }
    base = _FakeBase(plan)
    llm = _StubLLM(["sub A\nsub B\nsub C"])  # 3 sub-queries
    agent = AgenticRetriever(base=base, llm=llm, model="m", max_subqueries=4)
    out = await agent.retrieve(Query(text="q-original-ignored", top_k=4))
    # c2 ranks in all 3 lists → wins RRF over c1 (in only one list, even at rank 1).
    assert out[0].chunk_id == "c2"
    assert {r.chunk_id for r in out} == {"c1", "c2", "c3", "c4"}
    # Base called once per sub-query; never on the original (decomposed away).
    assert sorted(base.calls) == ["sub A", "sub B", "sub C"]


async def test_llm_failure_falls_back_to_base_on_original() -> None:
    base = _FakeBase({"q": [("c1", 0.9)]})
    llm = _StubLLM([], boom=True)
    agent = AgenticRetriever(base=base, llm=llm, model="m")
    out = await agent.retrieve(Query(text="q", top_k=5))
    assert [r.chunk_id for r in out] == ["c1"]
    assert base.calls == ["q"]  # graceful: single pass on original


async def test_max_subqueries_caps_fan_out() -> None:
    names = [f"sub query {i}" for i in range(6)]
    plan: dict[str, list[tuple[str, float]]] = {n: [("c1", 0.5)] for n in names}
    base = _FakeBase(plan)
    llm = _StubLLM(["\n".join(names)])  # 6 candidates
    agent = AgenticRetriever(base=base, llm=llm, model="m", max_subqueries=3)
    await agent.retrieve(Query(text="orig", top_k=5))
    assert len(base.calls) == 3  # capped to max_subqueries


async def test_top_k_passed_through_to_fused_output() -> None:
    names = [f"sub query {i}" for i in range(2)]
    plan = {n: [(f"c{j}", 0.5) for j in range(10)] for n in names}
    base = _FakeBase(plan)
    llm = _StubLLM(["\n".join(names)])
    agent = AgenticRetriever(base=base, llm=llm, model="m")
    out = await agent.retrieve(Query(text="orig", top_k=3))
    assert len(out) == 3
