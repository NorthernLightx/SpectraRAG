"""Settings → component wiring for the API.

Pure construction logic: given a ``Settings``, build and register the
generator / retriever / visual leg / classifier. Kept out of ``main.py`` so
the app factory there stays focused on app assembly (middleware, routers,
static mounts, instrumentation). ``create_app`` and the lifespan handler call
into the two ``_wire_*`` entry points here.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.api.deps import set_chunks, set_generator, set_retriever
from src.config.settings import Settings
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.embeddings.protocol import Embedder
from src.llm.ollama_chat import OllamaChatClient
from src.llm.openrouter import OpenRouterClient
from src.observability.logging import get_logger
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.generate import Generator
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.protocol import Retriever
from src.rag.retrievers.routing import RoutingRetriever
from src.rag.vectorstore import QdrantVectorStore

if TYPE_CHECKING:
    from src.rag.retrievers.classifier_llm import LLMQueryClassifier

# Mirrors the layout `scripts/eval_run.py` writes / `Generator._collect_image_paths`
# reads: `<pages_dir>/<paper_id>/<paper_id>_p<N>.png`. The paper id allows
# arbitrary characters except `/`, so we anchor on the trailing `_p<N>.png`.
_PAGE_FILE_RE = re.compile(r"^(?P<paper>.+)_p(?P<page>\d+)\.png$")


def _wire_generator_from_settings(settings: Settings) -> bool:
    """Build OpenRouterClient + Generator and register, when the API key is configured.

    Returns True if a Generator was wired, False if the key is unset (no-op).
    """
    if settings.openrouter_api_key is None:
        return False
    client = OpenRouterClient(api_key=settings.openrouter_api_key.get_secret_value())
    set_generator(
        Generator(
            llm=client,
            prompt=load_prompt_by_name("answer"),
            model=settings.default_chat_model,
            temperature=settings.temperature,
            max_context_tokens=settings.max_context_tokens,
            # When pages_dir is set the Generator attaches the rendered page PNG
            # for any visual RetrievalResult so a vision-capable default_chat_model
            # can read images directly. None = text-only behaviour (back-compat).
            pages_dir=settings.pages_dir,
            # Calibrated refusal gate (settings docstring + ADR 0009 follow-up).
            refusal_score_threshold=settings.refusal_score_threshold,
        )
    )
    return True


def _collect_pages_from_dir(pages_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Scan a `pages_dir` populated by ingestion into the `pages_by_paper` shape
    that `build_visual_retriever` consumes. Layout: each paper id maps to a
    subdirectory containing `<paper_id>_p<N>.png`. Returns an empty dict when
    the directory is missing or contains no matching PNGs (no exception — the
    caller treats that as "skip the visual leg")."""
    pages: dict[str, list[tuple[int, Path]]] = {}
    if not pages_dir.exists() or not pages_dir.is_dir():
        return pages
    for paper_subdir in sorted(pages_dir.iterdir()):
        if not paper_subdir.is_dir():
            continue
        paper_id = paper_subdir.name
        page_files: list[tuple[int, Path]] = []
        for png in sorted(paper_subdir.glob("*.png")):
            match = _PAGE_FILE_RE.match(png.name)
            if match is None or match.group("paper") != paper_id:
                continue
            page_files.append((int(match.group("page")), png))
        if page_files:
            page_files.sort(key=lambda pair: pair[0])
            pages[paper_id] = page_files
    return pages


async def _build_visual_retriever_from_settings(settings: Settings) -> Retriever | None:
    """Build the visual leg (ColQwen2 multivector index) when prerequisites
    are met. Returns None on any failure path: no `pages_dir`, empty layout,
    GPU/CPU OOM, missing colpali deps. The caller logs and falls back to
    text-only routing (the strong baseline per ADR 0008).

    Heavy import of ``torch`` / ``colpali_engine`` is deferred to keep the
    text-only deploy path light.
    """
    log = get_logger(__name__)
    if settings.pages_dir is None:
        log.info("api.multimodal.visual.skip_no_pages_dir")
        return None
    pages_by_paper = _collect_pages_from_dir(settings.pages_dir)
    if not pages_by_paper:
        log.warning("api.multimodal.visual.skip_empty_layout", pages_dir=str(settings.pages_dir))
        return None
    try:
        import torch

        from src.rag.retrievers.visual import build_visual_retriever

        device = "cuda" if torch.cuda.is_available() else "cpu"
        retriever = await build_visual_retriever(
            pages_by_paper, model_name=settings.visual_model, device=device
        )
        log.info(
            "api.multimodal.visual.wired",
            n_papers=len(pages_by_paper),
            n_pages=sum(len(v) for v in pages_by_paper.values()),
            device=device,
            model=settings.visual_model,
        )
        return retriever
    except Exception as exc:
        log.warning(
            "api.multimodal.visual.wire_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            model=settings.visual_model,
        )
        return None


def _build_classifier_from_settings(settings: Settings) -> LLMQueryClassifier | None:
    """Build the LLM query classifier. With an OpenRouter key it uses
    ``classifier_model`` over OpenRouter; without one it uses
    ``classifier_ollama_model`` over local Ollama. ADR 0013: the Ollama
    path measured +10.8 % recall@10 over the regex router on MMLongBench
    (~80 % of the oracle ceiling), so a keyless deploy no longer degrades
    to the weak regex classifier. Falls back to None (regex) only on hard
    failure — the regex stays the safe default. ADR 0008 §"Decision" §1:
    misclassification is bounded.
    """
    log = get_logger(__name__)
    try:
        from src.rag.retrievers.classifier_llm import LLMQueryClassifier

        prompt = load_prompt_by_name("classify_query")
        if settings.openrouter_api_key is not None:
            return LLMQueryClassifier(
                llm=OpenRouterClient(api_key=settings.openrouter_api_key.get_secret_value()),
                model=settings.classifier_model,
                prompt=prompt,
            )
        return LLMQueryClassifier(
            llm=OllamaChatClient(base_url=settings.ollama_base_url),
            model=settings.classifier_ollama_model,
            prompt=prompt,
        )
    except Exception as exc:
        log.warning(
            "api.multimodal.classifier.wire_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


async def _wire_retriever_from_settings(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vectorstore: QdrantVectorStore | None = None,
    visual_retriever: Retriever | None = None,
    classifier: LLMQueryClassifier | None = None,
) -> bool:
    """Materialise the production retriever from the configured Qdrant corpus.

    Connects to Qdrant, scrolls the configured collection so BM25 + chunks_by_id
    can be rebuilt in process, then registers a retriever via ``set_retriever``.
    Returns True on success, False if Qdrant is unreachable, the collection is
    empty, or the payload schema is stale. The API still serves health +
    OpenAPI; /answer just returns 503 until a corpus exists.

    When ``settings.enable_multimodal`` is True the function additionally
    attempts to build the visual leg (ColQwen2 over ``pages_dir``) and the
    LLM classifier (over ``openrouter_api_key``). If the visual leg builds,
    the registered retriever is a ``RoutingRetriever`` wrapping the text leg
    + visual leg + classifier — figure/table/multi_hop queries dispatch to
    RRF-fused hybrid, factual/definitional stay text-only. If the visual leg
    can't be built (no GPU, missing pages, model load error) the function
    falls through to text-only — same as ``enable_multimodal=False``.

    The keyword args allow tests to inject fakes without monkeypatching
    module-level constructors. Production callers pass none.
    """
    log = get_logger(__name__)
    try:
        if embedder is None:
            if settings.embedder_backend == "sentence_transformers":
                # Both the torch/sentence-transformers import (deferred to keep
                # it off the local-dev hot path where Ollama is the default)
                # and the constructor's ~2 GB bge-m3 weight load are
                # synchronous and slow. Run the whole thing off-thread so it
                # can't stall the event loop while the lifespan background
                # wiring task runs — otherwise /health and the static demo
                # would hang for the duration of import + load on cold start.
                def _build_st_embedder() -> Embedder:
                    from src.embeddings.sentence_transformers_bge import (
                        SentenceTransformersBgeEmbedder,
                    )

                    return SentenceTransformersBgeEmbedder()

                embedder = await asyncio.to_thread(_build_st_embedder)
            else:
                embedder = OllamaBgeEmbedder(base_url=settings.ollama_base_url)
        if vectorstore is None:
            vectorstore = QdrantVectorStore(
                url=settings.qdrant_url,
                collection_name=settings.corpus_collection,
                dim=embedder.dim,
            )
        chunks = await vectorstore.scroll_chunks()
    except Exception as exc:
        log.warning(
            "api.retriever.wire_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            qdrant_url=settings.qdrant_url,
            collection=settings.corpus_collection,
        )
        return False
    if not chunks:
        log.info(
            "api.retriever.skip_empty_corpus",
            qdrant_url=settings.qdrant_url,
            collection=settings.corpus_collection,
        )
        return False
    bm25 = Bm25Index()
    bm25.add(chunks)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    set_chunks(chunks_by_id)
    text_retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        candidate_pool=settings.rerank_top_k,
        # ADR 0014: the API ran unreranked while every eval/baseline reranks, so
        # the live system never delivered the measured retrieval quality.
        # `reranker_model` defaults to the baseline's bge-reranker-v2-m3 +
        # length-norm (ADR 0009); the CPU-only Cloud Run deploy overrides it to a
        # small MiniLM cross-encoder (RAG_RERANKER_MODEL) because the 568M bge
        # model reranks the pool in minutes per query without a GPU.
        reranker=BgeReranker(model_name=settings.reranker_model, length_norm=True),
        exclude_decoration=settings.exclude_decoration_chunks,
    )

    if settings.enable_multimodal:
        if visual_retriever is None:
            visual_retriever = await _build_visual_retriever_from_settings(settings)
        if classifier is None:
            classifier = _build_classifier_from_settings(settings)
        if visual_retriever is not None:
            set_retriever(
                RoutingRetriever(
                    text=text_retriever,
                    visual=visual_retriever,
                    classifier=classifier,
                )
            )
            log.info(
                "api.retriever.wired",
                mode="routing",
                chunks=len(chunks),
                classifier="llm" if classifier is not None else "regex",
            )
            return True
        log.info(
            "api.retriever.multimodal_degraded_to_text",
            reason="visual_leg_unavailable",
        )

    set_retriever(text_retriever)
    log.info(
        "api.retriever.wired",
        mode="text",
        qdrant_url=settings.qdrant_url,
        collection=settings.corpus_collection,
        chunks=len(chunks),
    )
    return True
