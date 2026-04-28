"""PipelineRetriever: hybrid (BM25 + dense) retrieval with RRF fusion and optional rerank."""

from __future__ import annotations

from src.embeddings.protocol import Embedder
from src.observability.logging import get_logger, timed_event
from src.rag.bm25 import Bm25Index
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.rerank import BgeReranker
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Query, RetrievalResult

_log = get_logger(__name__)


class PipelineRetriever:
    """Hybrid retriever: dense (Qdrant) + sparse (BM25), fused via RRF, optionally reranked."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vectorstore: QdrantVectorStore,
        bm25: Bm25Index,
        chunks_by_id: dict[str, Chunk],
        candidate_pool: int = 50,
        reranker: BgeReranker | None = None,
        rerank_input_size: int = 50,
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._bm25 = bm25
        self._chunks_by_id = chunks_by_id
        self._candidate_pool = candidate_pool
        self._reranker = reranker
        self._rerank_input_size = rerank_input_size

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        with timed_event(_log, "retrieve.done", query=query.text, top_k=query.top_k) as ctx:
            [vector] = await self._embedder.embed_texts([query.text])
            dense_hits = await self._vectorstore.search(vector, top_k=self._candidate_pool)
            sparse_hits = self._bm25.search(query.text, top_k=self._candidate_pool)
            ctx["dense_hits"] = len(dense_hits)
            ctx["sparse_hits"] = len(sparse_hits)
            ctx["candidate_pool"] = self._candidate_pool

            dense_ranked = [RankedItem(id=h.chunk_id, score=h.score) for h in dense_hits]
            sparse_ranked = [RankedItem(id=h.chunk_id, score=h.score) for h in sparse_hits]

            if self._reranker is not None:
                rrf_top = reciprocal_rank_fusion(
                    [dense_ranked, sparse_ranked], top_k=self._rerank_input_size
                )
                chunks_to_rerank = [
                    self._chunks_by_id[item.id]
                    for item in rrf_top
                    if item.id in self._chunks_by_id
                ]
                reranked = self._reranker.rerank(query.text, chunks_to_rerank, top_k=query.top_k)
                results = [
                    self._make_result(self._chunks_by_id[hit.chunk_id], hit.rerank_score)
                    for hit in reranked
                    if hit.chunk_id in self._chunks_by_id
                ]
                ctx["stage"] = "rerank"
                ctx["returned"] = len(results)
                ctx["top_chunk"] = results[0].chunk_id if results else None
                return results

            fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_k=query.top_k)
            results = [
                self._make_result(self._chunks_by_id[item.id], item.score)
                for item in fused
                if item.id in self._chunks_by_id
            ]
            ctx["stage"] = "rrf"
            ctx["returned"] = len(results)
            ctx["top_chunk"] = results[0].chunk_id if results else None
            return results

    def _make_result(self, chunk: Chunk, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            score=score,
            text=chunk.text,
            page_numbers=chunk.page_numbers,
            source="pipeline",
            metadata={"section": chunk.section} if chunk.section else {},
        )
