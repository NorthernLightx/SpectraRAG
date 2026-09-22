"""Health endpoint: confirms the service is up and reports build/env."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import APIRouter, Depends

from src.api.deps import get_settings, peek_retrieval_config, peek_retriever
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
    to know about up front. One is whether page images are served at /pages/;
    the BYOK client uses this to decide if it should attach image content
    blocks in its OpenRouter call."""
    pages_available = settings.pages_dir is not None and settings.pages_dir.is_dir()
    wired = peek_retrieval_config()
    return {
        "status": "ok",
        "version": _service_version(),
        "env": settings.env,
        "pages_available": pages_available,
        # Whether the multimodal router is live. False when the visual leg
        # couldn't build (a CPU-only deploy, say) and force_route/routing_mode
        # are no-ops; the UI greys those controls out instead of letting them
        # silently do nothing.
        "routing_available": isinstance(peek_retriever(), RoutingRetriever),
        # Whether POST /ingest accepts uploads (ADR 0029). The Papers tab shows
        # its "Add PDF" control only when this is true.
        "upload_available": settings.enable_upload,
        # What the wired retriever actually runs. The fingerprint equals the
        # `retrieval_fingerprint` of any eval run that measured this stack.
        "retrieval": None
        if wired is None
        else {
            "profile": settings.profile,
            "fingerprint": wired.fingerprint(),
            "config": wired.as_dict(),
        },
    }
