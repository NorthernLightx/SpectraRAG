"""Health endpoint: confirms the service is up and reports build/env."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import APIRouter, Depends

from src.api.deps import get_settings, peek_retriever
from src.config.settings import Settings
from src.rag.retrievers.routing import RoutingRetriever

router = APIRouter()


def _service_version() -> str:
    try:
        return version("spectrarag")
    except PackageNotFoundError:
        return "0.0.0+local"


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Returns liveness + the small set of feature flags the bundled UI needs
    to know about up front (whether page images are served at /pages/ — the
    BYOK client uses this to decide if it should attach image content blocks
    in its OpenRouter call)."""
    pages_available = settings.pages_dir is not None and settings.pages_dir.is_dir()
    return {
        "status": "ok",
        "version": _service_version(),
        "env": settings.env,
        "pages_available": pages_available,
        # Whether /demo/chat can generate (ADR 0027). The UI uses this to
        # decide between the keyless demo path and the BYOK-only notice.
        "demo_available": settings.demo_openrouter_key is not None and settings.demo_daily_cap > 0,
        # Whether the multimodal router is live. False when the visual leg
        # couldn't build (e.g. CPU-only deploy) and force_route/routing_mode
        # are no-ops; the UI greys those controls out instead of letting them
        # silently do nothing.
        "routing_available": isinstance(peek_retriever(), RoutingRetriever),
    }
