"""Runtime configuration: Pydantic Settings layered over a YAML default."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"
_ENV_PREFIX = "RAG_"


class Settings(BaseSettings):
    """Runtime configuration. RAG_-prefixed env vars override YAML.

    Secrets are deliberately omitted in Phase 1 day-one scaffolding — no client
    actually consumes one yet. They will be added (still RAG_-prefixed) when
    the first wired client needs them.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "local"
    log_level: str = "INFO"

    default_chat_model: str = "anthropic/claude-3.5-sonnet"
    default_embed_model: str = "bge-m3"

    top_k: int = Field(default=5, ge=1)
    rerank_top_k: int = Field(default=50, ge=1)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    max_context_tokens: int = Field(default=8000, ge=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    langfuse_host: str = "http://localhost:3000"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config YAML at {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings: YAML defaults < env vars.

    Pydantic Settings normally treats constructor kwargs as highest priority,
    which would invert the precedence we want. So we strip any YAML key whose
    matching `RAG_*` env var is already set — letting env vars win.
    """
    yaml_values = _read_yaml(config_path or DEFAULT_CONFIG_PATH)
    overrides = {
        key: value
        for key, value in yaml_values.items()
        if f"{_ENV_PREFIX}{key.upper()}" not in os.environ
    }
    return Settings(**overrides)
