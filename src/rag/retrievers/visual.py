"""VisualRetriever: ColQwen2 multi-vector page retrieval.

Each page of each paper is embedded once into a multi-vector tensor
(`[n_patches, dim]`). At query time the user query is also multi-vector
embedded and scored against every page via late interaction
(MaxSim: for each query token, max similarity over page patches, summed
across query tokens).

Storage is in-memory (a list of tensors keyed by chunk_id). For a small
multi-paper corpus (~5 papers x 20-80 pages) that's tens of MB at bf16. If
we ever want to persist these, Qdrant 1.10+ supports multivector collections
natively.

Duck-types `Retriever` so `eval.runner.evaluate` can run it head-to-head
against `PipelineRetriever`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from src.observability.logging import get_logger, timed_event
from src.types import Query, RetrievalResult

if TYPE_CHECKING:
    from src.rag.visual_store import QdrantVisualStore

_log = get_logger(__name__)

# Sentinel chunk-id format mirrors text-path chunks: `<paper>::p<n>::page`.
# Distinct from `::cN` so visual page chunks never collide with text chunks.
_PAGE_CHUNK_FMT = "{paper_id}::p{page_no}::page"


class VisualRetriever:
    """ColQwen2-backed multi-vector retriever.

    Construct once via `build_visual_retriever(...)` (async, runs the
    embedding pass) - or use the lighter `VisualRetriever(model, processor,
    page_embeds)` constructor when you already have embeddings cached.
    """

    def __init__(
        self,
        *,
        model: Any,  # colpali ColQwen2; typed Any to keep colpali optional at import time
        processor: Any,
        page_embeds: dict[str, torch.Tensor] | None = None,
        page_meta: dict[str, tuple[str, int]] | None = None,
        store: QdrantVisualStore | None = None,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._processor = processor
        self._page_embeds = page_embeds or {}
        self._page_meta = page_meta or {}
        # When set, page vectors live in a persisted Qdrant multivector
        # collection and scoring runs there (the deploy path); page_embeds is
        # then empty. When None, scoring is in-memory over page_embeds (the
        # offline/eval path). ADR 0028.
        self._store = store
        self._device = device

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        """Embed the query and rank pages by MaxSim.

        Two scoring backends: a persisted Qdrant multivector collection
        (``self._store``, the deploy path) or the in-memory ``page_embeds``
        (offline/eval). Either way only the query is embedded here; page
        vectors were embedded ahead of time. ADR 0009 follow-up: the paper-id
        filter from ``query.filters['paper_id']`` is honored on both backends.
        """
        paper_filter = query.paper_id_filter()

        if self._store is not None:
            with timed_event(
                _log, "visual_retrieve.done", query=query.text, top_k=query.top_k, backend="qdrant"
            ) as ctx:
                query_vec = await asyncio.to_thread(self._embed_query, query.text)
                hits = await self._store.search(query_vec, query.top_k, paper_filter=paper_filter)
                ctx["paper_filter"] = paper_filter or ""
                ctx["returned"] = len(hits)
                ctx["top_chunk"] = hits[0].chunk_id if hits else None
                return hits

        if not self._page_embeds:
            return []

        # The in-memory index is `page_meta[chunk_id] = (paper_id, page_no)`,
        # so the paper filter is a dict-key check before scoring.
        with timed_event(_log, "visual_retrieve.done", query=query.text, top_k=query.top_k) as ctx:
            scores = await asyncio.to_thread(self._score_query, query.text)
            if paper_filter is not None:
                scores = {
                    cid: s for cid, s in scores.items() if self._page_meta[cid][0] == paper_filter
                }
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: query.top_k]
            results: list[RetrievalResult] = []
            for chunk_id, score in ranked:
                paper_id, page_no = self._page_meta[chunk_id]
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        score=float(score),
                        text=f"[Page image {paper_id} p{page_no}]",
                        page_numbers=[page_no],
                        source="visual",
                        score_kind="maxsim",
                    )
                )
            ctx["candidate_pool"] = len(self._page_embeds)
            ctx["paper_filter"] = paper_filter or ""
            ctx["returned"] = len(results)
            ctx["top_chunk"] = results[0].chunk_id if results else None
            return results

    def _encode_query(self, query: str) -> torch.Tensor:
        """Embed a query string to its ``[1, n_q_tokens, dim]`` multivector.
        Shared by the in-memory MaxSim path and the Qdrant store path."""
        with torch.no_grad():
            batch_q = self._processor.process_queries([query]).to(self._device)
            embed: torch.Tensor = self._model(**batch_q)  # [1, n_q_tokens, dim]
            return embed

    def _score_query(self, query: str) -> dict[str, float]:
        """Synchronous: embed query, score MaxSim against every page in one batched call.

        Earlier this looped per-page which meant ~200 GPU sync calls per query;
        batching all pages into a single `score_multi_vector` invocation drops
        per-query latency by ~5x with no semantic change (MaxSim is stateless).
        Pages have variable patch counts so we pad to the max length and rely
        on the score function's masking, and colpali-engine handles ragged inputs
        when given a list of tensors.
        """
        query_embed = self._encode_query(query)
        chunk_ids = list(self._page_embeds.keys())
        page_tensors = [self._page_embeds[cid] for cid in chunk_ids]
        with torch.no_grad():
            # `score_multi_vector` accepts a list of [n_p, dim] tensors and
            # returns a [B_q, B_p] similarity matrix. One-shot; no Python loop.
            scores_matrix = self._processor.score_multi_vector(query_embed, page_tensors)
        # query_embed batch dim = 1, so squeeze it; result is [B_p].
        row = scores_matrix.squeeze(0).float().cpu()
        return dict(zip(chunk_ids, row.tolist(), strict=True))

    def _embed_query(self, query: str) -> list[list[float]]:
        """Embed a query to its ``[n_q_tokens, dim]`` multivector as a list of
        rows, the 2D query the Qdrant MAX_SIM comparator expects."""
        squeezed: torch.Tensor = self._encode_query(query).squeeze(0).float().cpu()
        rows: list[list[float]] = squeezed.tolist()
        return rows


def _select_col_classes(model_name: str) -> tuple[Any, Any]:
    """Pick the colpali-engine model + processor pair for a given HF model id.

    The colpali-engine library has separate classes per backbone family
    (ColQwen2 vs ColQwen2_5 vs ColPali). We dispatch on the model name so
    callers can swap backbones via config without touching this file."""
    from colpali_engine.models import (
        ColPali,
        ColPaliProcessor,
        ColQwen2,
        ColQwen2_5,
        ColQwen2_5_Processor,
        ColQwen2Processor,
        ColQwen3,
        ColQwen3_5,
        ColQwen3_5Processor,
        ColQwen3Processor,
    )

    # Order matters: the point-release names contain the base name as a
    # substring, so the more specific prefix has to be tested first.
    name = model_name.lower()
    if "colqwen3.5" in name:
        return ColQwen3_5, ColQwen3_5Processor
    if "colqwen3" in name:
        return ColQwen3, ColQwen3Processor
    if "colqwen2.5" in name:
        return ColQwen2_5, ColQwen2_5_Processor
    if "colqwen2" in name:
        return ColQwen2, ColQwen2Processor
    if "colpali" in name:
        return ColPali, ColPaliProcessor
    raise ValueError(
        f"unsupported visual model {model_name!r}; expected a vidore/colqwen2*, "
        "vidore/colqwen2.5*, vidore/colqwen3*, or vidore/colpali* checkpoint"
    )


async def load_visual_model(
    model_name: str, device: str, *, dtype: torch.dtype | None = None
) -> tuple[Any, Any]:
    """Load a Col* model + processor (no page embedding).

    Shared by the serve path (which encodes only the query against a persisted
    Qdrant page index, ADR 0028) and by ``build_visual_retriever``, which then
    embeds pages in-process. Defaults to bf16 on GPU and fp32 on CPU (bf16
    matmuls are not uniformly supported there); `dtype` overrides it.
    """
    model_cls, processor_cls = _select_col_classes(model_name)
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    def _load() -> tuple[Any, Any]:
        m = model_cls.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
        m.train(False)  # set inference mode
        p = processor_cls.from_pretrained(model_name)
        return m, p

    return await asyncio.to_thread(_load)


async def build_visual_retriever(
    pages_by_paper: dict[str, list[tuple[int, Path]]],
    *,
    model_name: str = "vidore/colqwen2-v1.0",
    device: str = "cuda",
) -> VisualRetriever:
    """Load a Col* visual retriever, embed every supplied page, return it.

    `pages_by_paper` maps `paper_id -> [(page_number, image_path), ...]`.
    Embedding runs in a worker thread via `asyncio.to_thread` so the caller's
    event loop isn't blocked on the GPU work. The model class is picked from
    the name. Default is ColQwen2-v1.0 (Qwen2-VL-2B backbone, ~4 GB VRAM at
    bf16). ColQwen2.5-v0.2 (Qwen2.5-VL-3B, ~6 GB) is the 2025 upgrade and
    works on hardware with ≥7 GB free GPU; on an 8 GB consumer card with a
    Windows desktop compositor + Ollama runtime holding ~3 GB, the bigger
    model OOMs. Pass `--model vidore/colqwen2.5-v0.2` on a roomier GPU.
    """
    model, processor = await load_visual_model(model_name, device)

    page_embeds: dict[str, torch.Tensor] = {}
    page_meta: dict[str, tuple[str, int]] = {}

    def _embed_page(image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        with torch.no_grad():
            batch = processor.process_images([rgb]).to(device)
            embed = model(**batch)  # [1, n_patches, dim]
        squeezed: torch.Tensor = embed.squeeze(0)  # [n_patches, dim]
        return squeezed

    with timed_event(
        _log,
        "visual_index.done",
        n_papers=len(pages_by_paper),
        model=model_name,
        device=device,
    ) as ctx:
        n_pages = 0
        n_failed = 0
        for paper_id, page_list in pages_by_paper.items():
            for page_no, image_path in page_list:
                chunk_id = _PAGE_CHUNK_FMT.format(paper_id=paper_id, page_no=page_no)
                try:
                    embed = await asyncio.to_thread(_embed_page, image_path)
                except (RuntimeError, OSError, ValueError) as exc:
                    n_failed += 1
                    _log.warning(
                        "visual_embed.skip",
                        paper_id=paper_id,
                        page=page_no,
                        path=str(image_path),
                        error=str(exc),
                    )
                    continue
                page_embeds[chunk_id] = embed
                page_meta[chunk_id] = (paper_id, page_no)
                n_pages += 1
        ctx["n_pages"] = n_pages
        ctx["n_failed"] = n_failed

    return VisualRetriever(
        model=model,
        processor=processor,
        page_embeds=page_embeds,
        page_meta=page_meta,
        device=device,
    )
