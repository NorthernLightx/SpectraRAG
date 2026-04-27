"""FastAPI dependency-injection wiring."""

from __future__ import annotations

from functools import lru_cache

from src.config.settings import Settings, load_settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor for FastAPI dependencies."""
    return load_settings()
