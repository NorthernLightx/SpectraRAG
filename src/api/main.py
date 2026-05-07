"""FastAPI app factory."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.auth import make_api_key_middleware
from src.api.deps import set_generator, set_retriever
from src.api.middleware import request_context_middleware
from src.api.rate_limit import limiter
from src.api.routes import answer, health, query
from src.config.settings import Settings, load_settings
from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.embeddings.protocol import Embedder
from src.llm.openrouter import OpenRouterClient
from src.observability.logging import configure_logging, get_logger
from src.observability.otel import configure_otel
from src.observability.sentry import configure_sentry
from src.prompts.loader import load_prompt_by_name
from src.rag.bm25 import Bm25Index
from src.rag.generate import Generator
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
    """Build the LLM query classifier when an OpenRouter key is set. Falls
    back to None (regex classifier) on any failure — the regex is the safe
    default. ADR 0008 §"Decision" §1: misclassification is bounded.
    """
    log = get_logger(__name__)
    if settings.openrouter_api_key is None:
        log.info("api.multimodal.classifier.skip_no_api_key")
        return None
    try:
        from src.rag.retrievers.classifier_llm import LLMQueryClassifier

        client = OpenRouterClient(api_key=settings.openrouter_api_key.get_secret_value())
        return LLMQueryClassifier(
            llm=client,
            model=settings.classifier_model,
            prompt=load_prompt_by_name("classify_query"),
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
                # Deferred import keeps the heavy torch/sentence-transformers
                # import off the local-dev hot path where Ollama is the default.
                from src.embeddings.sentence_transformers_bge import (
                    SentenceTransformersBgeEmbedder,
                )

                embedder = SentenceTransformersBgeEmbedder()
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
    text_retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        candidate_pool=settings.rerank_top_k,
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


def create_app(*, log_file: Path | None = Path("logs/api.log")) -> FastAPI:
    settings = load_settings()
    configure_logging(level=settings.log_level, env=settings.env, log_file=log_file)
    log = get_logger(__name__)

    sentry_on = configure_sentry()
    otel_on = configure_otel()
    generator_on = _wire_generator_from_settings(settings)
    log.info(
        "api.startup",
        env=settings.env,
        log_level=settings.log_level,
        sentry=sentry_on,
        otel=otel_on,
        generator=generator_on,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Retriever wiring needs an event loop (Qdrant scroll is async), so it
        # runs from the lifespan handler rather than the sync factory. Tests
        # that construct ``TestClient(app)`` without ``with`` skip lifespan and
        # rely on ``dependency_overrides`` for an injected retriever — that's
        # exactly the pre-existing pattern, no test changes needed.
        retriever_on = await _wire_retriever_from_settings(settings)
        log.info("api.lifespan.startup", retriever=retriever_on)
        yield

    app = FastAPI(
        title="Multi-modal Paper RAG",
        version="0.1.0",
        description="RAG over scientific papers comparing pipeline vs visual retrieval.",
        lifespan=lifespan,
    )
    # slowapi needs:
    #  1. limiter on app.state (read by the @limiter.limit decorator on routes)
    #  2. a handler for RateLimitExceeded so it returns 429 instead of 500
    #  3. SlowAPIMiddleware — without it the rate check fires AFTER Depends
    #     resolution, so endpoint-level guards (e.g. the unset-retriever 503)
    #     short-circuit before the limiter counts the request and the bucket
    #     never fills. Middleware moves the check above the Depends chain.
    # The type-ignore is the standard slowapi workaround: Starlette types the
    # handler arg as Exception but slowapi narrows to RateLimitExceeded —
    # covariant in practice, mypy strict can't see across the inheritance.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)
    app.middleware("http")(request_context_middleware)
    # Auth runs OUTERMOST so unauthenticated requests get short-circuited
    # before request_context allocates an X-Request-ID or downstream code does
    # any work. Pass None when no key is configured — the middleware no-ops
    # and the endpoint-level guards take over.
    api_key = settings.public_api_key.get_secret_value() if settings.public_api_key else None
    app.middleware("http")(make_api_key_middleware(api_key))

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(answer.router)

    # Page PNGs served at /pages/<paper>/<paper>_pN.png. The browser pulls
    # these URLs into OpenRouter `image_url` content blocks so a vision-
    # capable model (gpt-4o, claude, qwen3-vl) sees the pixels directly. This
    # is the deploy-side equivalent of `Generator._collect_image_paths`'s
    # server-side attachment: same data, different transport. Mounted from
    # settings.pages_dir when set (defaults to None — no pages served).
    if settings.pages_dir is not None and settings.pages_dir.is_dir():
        app.mount("/pages", StaticFiles(directory=settings.pages_dir), name="pages")

    # Static frontend mounted LAST at "/" so it doesn't shadow API routes —
    # FastAPI matches explicit routes before mounted apps. `html=True` makes
    # GET / serve index.html (instead of a directory listing). When the web/
    # directory isn't present (e.g., a stripped runtime image) the mount
    # silently skips so the API still boots.
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    # Auto-instrumentation must run after routers are added so per-route
    # spans are named correctly. HTTPXClientInstrumentor is a singleton
    # and BaseInstrumentor.instrument() is internally idempotent, so a
    # repeat call is a no-op (logs a warning, no exception).
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return app


_log_file: Path | None = None if os.getenv("RAG_ENV") == "prod" else Path("logs/api.log")
app = create_app(log_file=_log_file)
