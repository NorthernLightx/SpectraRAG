"""FastAPI app factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.api.deps import set_generator
from src.api.middleware import request_context_middleware
from src.api.routes import answer, health, query
from src.config.settings import Settings, load_settings
from src.llm.openrouter import OpenRouterClient
from src.observability.logging import configure_logging, get_logger
from src.observability.otel import configure_otel
from src.observability.sentry import configure_sentry
from src.prompts.loader import load_prompt_by_name
from src.rag.generate import Generator


def _wire_generator_from_settings(settings: Settings) -> bool:
    """Build OpenRouterClient + Generator and register, when the API key is configured.

    Returns True if a Generator was wired, False if the key is unset (no-op).
    Retriever wiring is intentionally deferred — it needs a populated Qdrant
    collection, which is Phase 2's ingestion path. Eval scripts and tests
    continue to wire their own retrievers via ``set_retriever``.
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
        )
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

    app = FastAPI(
        title="Multi-modal Paper RAG",
        version="0.1.0",
        description="RAG over scientific papers comparing pipeline vs visual retrieval.",
    )
    app.middleware("http")(request_context_middleware)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "Multi-modal Paper RAG", "docs": "/docs"}

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(answer.router)

    # Auto-instrumentation must run after routers are added so per-route
    # spans are named correctly. HTTPXClientInstrumentor is a singleton
    # and BaseInstrumentor.instrument() is internally idempotent, so a
    # repeat call is a no-op (logs a warning, no exception).
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return app


_log_file: Path | None = None if os.getenv("RAG_ENV") == "prod" else Path("logs/api.log")
app = create_app(log_file=_log_file)
