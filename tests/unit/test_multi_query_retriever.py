"""MultiQueryRetriever: variant gathering + RRF fusion + base-retriever wrapping."""

from __future__ import annotations

from src.llm.protocol import ChatResponse, Message
from src.rag.query_expansion import QueryExpander
from src.rag.retrievers.multi_query import MultiQueryRetriever
from src.types import Query, RetrievalResult


class _StubLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    async def chat(self, messages: list[Message], model: str, **kwargs: object) -> ChatResponse:
        # Record the user message content for debugging.
        self.calls.append(messages[-1].content if messages else "")
        text = self._replies.pop(0) if self._replies else ""
        return ChatResponse(text=text, model=model, tokens_in=1, tokens_out=1)


class _FakeRetriever:
    """Returns a deterministic, query-keyed list — used to verify that
    MultiQueryRetriever calls the base for each variant."""

    def __init__(self, plan: dict[str, list[tuple[str, float]]]) -> None:
        self._plan = plan
        self.calls: list[str] = []

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.calls.append(query.text)
        results = self._plan.get(query.text, [])
        return [_make_result(cid, score) for cid, score in results]


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id="p1",
        score=score,
        text=f"text for {chunk_id}",
        page_numbers=[1],
        source="pipeline",
    )


async def test_multi_query_rewrite_mode_fuses_variant_results() -> None:
    """Original query finds c1>c2; rewrite finds c2>c3. RRF should rank c2 first."""
    plan = {
        "What is X?": [("c1", 0.9), ("c2", 0.5)],
        "How is X defined?": [("c2", 0.95), ("c3", 0.7)],
        "What does X mean?": [("c2", 0.8), ("c1", 0.4)],
    }
    base = _FakeRetriever(plan)
    llm = _StubLLM(["How is X defined?\nWhat does X mean?"])
    expander = QueryExpander(llm=llm, model="m")
    mq = MultiQueryRetriever(base=base, expander=expander, mode="rewrite", n_rewrites=2)

    out = await mq.retrieve(Query(text="What is X?", top_k=3))

    assert base.calls == ["What is X?", "How is X defined?", "What does X mean?"]
    # c2 appears at rank 2/1/1 → highest cumulative RRF score
    assert next(r.chunk_id for r in out) == "c2"


async def test_multi_query_hyde_mode_uses_hyde_passage_as_variant() -> None:
    plan = {
        "What is X?": [("c1", 0.5)],
        "X is a method that achieves Y by computing Z.": [("c5", 0.95)],
    }
    base = _FakeRetriever(plan)
    llm = _StubLLM(["X is a method that achieves Y by computing Z."])
    expander = QueryExpander(llm=llm, model="m")
    mq = MultiQueryRetriever(base=base, expander=expander, mode="hyde")

    out = await mq.retrieve(Query(text="What is X?", top_k=3))

    assert "X is a method that achieves Y by computing Z." in base.calls
    chunk_ids = {r.chunk_id for r in out}
    assert "c5" in chunk_ids


async def test_multi_query_combo_mode_uses_rewrites_and_hyde() -> None:
    plan = {
        "Q?": [("c1", 0.5)],
        "Rephrased one?": [("c2", 0.9)],
        "Rephrased two?": [("c3", 0.8)],
        "A hypothetical answer passage.": [("c4", 0.7)],
    }
    base = _FakeRetriever(plan)
    llm = _StubLLM(["Rephrased one?\nRephrased two?", "A hypothetical answer passage."])
    expander = QueryExpander(llm=llm, model="m")
    mq = MultiQueryRetriever(base=base, expander=expander, mode="combo", n_rewrites=2)

    out = await mq.retrieve(Query(text="Q?", top_k=4))

    # All four queries hit the base retriever.
    assert set(base.calls) == {
        "Q?",
        "Rephrased one?",
        "Rephrased two?",
        "A hypothetical answer passage.",
    }
    # All four chunks land in top_k=4.
    assert {r.chunk_id for r in out} == {"c1", "c2", "c3", "c4"}


async def test_multi_query_falls_back_to_original_when_expander_returns_nothing() -> None:
    plan = {"What is X?": [("c1", 0.5), ("c2", 0.3)]}
    base = _FakeRetriever(plan)
    llm = _StubLLM([""])  # empty rewrite response
    expander = QueryExpander(llm=llm, model="m")
    mq = MultiQueryRetriever(base=base, expander=expander, mode="rewrite", n_rewrites=2)

    out = await mq.retrieve(Query(text="What is X?", top_k=2))

    assert base.calls == ["What is X?"]
    assert [r.chunk_id for r in out] == ["c1", "c2"]
