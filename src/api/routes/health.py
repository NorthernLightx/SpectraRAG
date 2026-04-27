"""Health endpoint: confirms the service is up and reports build/env."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends

from src.api.deps import get_settings
from src.config.settings import Settings

router = APIRouter()


def _service_version() -> str:
    try:
        return version("multi-modal-paper-rag")
    except PackageNotFoundError:
        return "0.0.0+local"


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "status": "ok",
        "version": _service_version(),
        "env": settings.env,
    }
