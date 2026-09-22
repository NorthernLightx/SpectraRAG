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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from src.api.deps import (
    set_chunks,
    set_corpus_handles,
    set_generator,
    set_retrieval_config,
    set_retriever,
)
from src.config.settings import Settings
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.embeddings.protocol import Embedder
from src.llm.ollama_chat import OllamaChatClient
from src.llm.openrouter import OpenRouterClient
from src.observability.logging import get_logger
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.generate import _MAX_VISION_IMAGES, Generator
from src.rag.retrieval_config import (
    RetrievalConfig,
    build_routing_retriever,
    build_text_retriever,
)
from src.rag.retrievers.protocol import Retriever
from src.rag.vectorstore import QdrantVectorStore

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

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
            # ADR 0024: when route-by-fit is enabled, a fitting whole document
            # resolves to ALL its pages; the per-call image cap must rise to the
            # page budget or _collect_image_paths silently truncates to 4 and
            # erases the win. Unset budget keeps the constructor default cap.
            max_vision_images=settings.page_budget or _MAX_VISION_IMAGES,
        )
    )
    return True


def _collect_pages_from_dir(pages_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Scan a `pages_dir` populated by ingestion into the `pages_by_paper` shape
    that `build_visual_retriever` consumes. Layout: each paper id maps to a
    subdirectory containing `<paper_id>_p<N>.png`. Returns an empty dict when
    the directory is missing or contains no matching PNGs (no exception; the
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


async def _build_visual_retriever_from_settings(
    settings: Settings, *, client: AsyncQdrantClient | None = None
) -> Retriever | None:
    """Build the visual leg from the persisted ColQwen2 page index (ADR 0028).

    Loads a ``QdrantVisualStore`` over ``settings.visual_collection`` and, when
    it holds pages, a ``VisualRetriever`` that scores against it, encoding only
    the query at serve time, with no startup page-encode. Returns None on any
    failure path: an empty or absent collection, GPU/CPU OOM, missing colpali
    deps. The caller logs and falls back to text-only routing (the strong
    baseline per ADR 0008).

    The store check is cheap and torch-free; the heavy ``torch`` /
    ``colpali_engine`` import + model load is deferred until the collection is
    known to hold pages, keeping the text-only deploy path light.
    """
    log = get_logger(__name__)
    from src.rag.visual_store import QdrantVisualStore

    # Share the text store's client: embedded path-mode allows only one client
    # per on-disk path, and the text leg already holds it open (ADR 0028).
    store = QdrantVisualStore(
        url=settings.qdrant_url, collection_name=settings.visual_collection, client=client
    )
    try:
        n_pages = await store.count()
    except Exception as exc:
        log.warning(
            "api.multimodal.visual.store_unavailable",
            error=str(exc),
            error_type=type(exc).__name__,
            collection=settings.visual_collection,
        )
        return None
    if n_pages == 0:
        # Reaching here means enable_multimodal is on but the visual index is
        # empty/absent (the index didn't ship in the image, for example). Warn,
        # don't whisper: the deploy silently degrades to text-only otherwise.
        log.warning(
            "api.multimodal.visual.skip_empty_collection",
            collection=settings.visual_collection,
            detail="multimodal enabled but visual index empty/absent; serving text-only",
        )
        return None
    try:
        import torch

        from src.rag.retrievers.visual import VisualRetriever, load_visual_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = await load_visual_model(settings.visual_model, device)
        retriever = VisualRetriever(model=model, processor=processor, store=store, device=device)
        log.info(
            "api.multimodal.visual.wired",
            backend="qdrant",
            n_pages=n_pages,
            device=device,
            model=settings.visual_model,
            collection=settings.visual_collection,
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
    failure, and the regex stays the safe default. ADR 0008 §"Decision" §1:
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
    + visual leg + classifier, running in ``settings.routing_mode``. The
    default fuses both legs on every query (ADR 0032). If the visual leg can't
    be built (no GPU, missing pages, model load error) the function falls
    through to text-only, same as ``enable_multimodal=False``.

    The keyword args allow tests to inject fakes without monkeypatching
    module-level constructors. Production callers pass none.
    """
    log = get_logger(__name__)
    config = RetrievalConfig.from_settings(settings)
    try:
        if embedder is None:
            if settings.embedder_backend == "sentence_transformers":
                # Both the torch/sentence-transformers import (deferred to keep
                # it off the local-dev hot path where Ollama is the default)
                # and the constructor's ~2 GB bge-m3 weight load are
                # synchronous and slow. Run the whole thing off-thread so it
                # can't stall the event loop while the lifespan background
                # wiring task runs. Otherwise /health and the static demo
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
    # ADR 0029: expose the live index objects so POST /ingest can append a
    # document at runtime through the same embedder / store / bm25.
    set_corpus_handles(embedder, vectorstore, bm25)
    text_retriever = build_text_retriever(
        config,
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
    )

    if config.visual_model is not None:
        if visual_retriever is None:
            visual_retriever = await _build_visual_retriever_from_settings(
                settings, client=vectorstore.client
            )
        if classifier is None:
            classifier = _build_classifier_from_settings(settings)
        if visual_retriever is not None:
            if config.routing_mode == "category" and classifier is None:
                config = replace(config, classifier="regex")
            # ADR 0032. `cascade` without a threshold raises here rather than
            # starting in a mode the retriever cannot run.
            set_retriever(
                build_routing_retriever(
                    config, text=text_retriever, visual=visual_retriever, classifier=classifier
                )
            )
            set_retrieval_config(config)
            log.info(
                "api.retriever.wired",
                mode="routing",
                routing_mode=settings.routing_mode,
                chunks=len(chunks),
                classifier="llm" if classifier is not None else "regex",
                retrieval_fingerprint=config.fingerprint(),
            )
            return True
        log.warning(
            "api.retriever.multimodal_degraded_to_text",
            reason="visual_leg_unavailable",
        )

    set_retriever(text_retriever)
    config = config.text_only()
    set_retrieval_config(config)
    log.info(
        "api.retriever.wired",
        mode="text",
        qdrant_url=settings.qdrant_url,
        collection=settings.corpus_collection,
        chunks=len(chunks),
        retrieval_fingerprint=config.fingerprint(),
    )
    return True
