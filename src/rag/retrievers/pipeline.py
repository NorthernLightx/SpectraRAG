"""PipelineRetriever: hybrid (BM25 + dense) retrieval with RRF fusion. No reranker yet."""

from __future__ import annotations

from src.embeddings.protocol import Embedder
from src.rag.bm25 import Bm25Index
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Query, RetrievalResult


class PipelineRetriever:
    """Hybrid retriever: dense (Qdrant) + sparse (BM25), fused via RRF."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vectorstore: QdrantVectorStore,
        bm25: Bm25Index,
        chunks_by_id: dict[str, Chunk],
        candidate_pool: int = 50,
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._bm25 = bm25
        self._chunks_by_id = chunks_by_id
        self._candidate_pool = candidate_pool

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        [vector] = await self._embedder.embed_texts([query.text])
        dense_hits = await self._vectorstore.search(vector, top_k=self._candidate_pool)
        sparse_hits = self._bm25.search(query.text, top_k=self._candidate_pool)

        dense_ranked = [RankedItem(id=h.chunk_id, score=h.score) for h in dense_hits]
        sparse_ranked = [RankedItem(id=h.chunk_id, score=h.score) for h in sparse_hits]

        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_k=query.top_k)
        results: list[RetrievalResult] = []
        for fused_item in fused:
            chunk = self._chunks_by_id.get(fused_item.id)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=fused_item.score,
                    text=chunk.text,
                    page_numbers=chunk.page_numbers,
                    source="pipeline",
                    metadata={"section": chunk.section} if chunk.section else {},
                )
            )
        return results
