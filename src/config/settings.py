"""Runtime configuration: Pydantic Settings layered over a YAML default.

.env is loaded once in src/__init__.py before any os.environ.get() runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"
_ENV_PREFIX = "RAG_"


class Settings(BaseSettings):
    """Runtime configuration. RAG_-prefixed env vars override YAML."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "local"
    log_level: str = "INFO"

    default_chat_model: str = "anthropic/claude-sonnet-4.6"
    default_embed_model: str = "bge-m3"
    # ADR 0008: ColPali-family checkpoint for the visual leg of routing.
    # Default fits the 8 GB RTX 3070 dev box; the 3 B+ tier (ColQwen2.5-v0.2,
    # ColQwen3, etc.) needs ≥7 GB free GPU and is opt-in via this knob.
    visual_model: str = "vidore/colqwen2-v1.0"

    openrouter_api_key: SecretStr | None = None
    # Optional shared-secret gate for /answer + /query. Unset = no auth (dev
    # default). Set this in any deployed env so the LLM-spending endpoints
    # can't be hit by drive-by traffic. Health + OpenAPI metadata routes stay
    # exempt.
    public_api_key: SecretStr | None = None

    top_k: int = Field(default=5, ge=1)
    rerank_top_k: int = Field(default=50, ge=1)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    # ADR 0008: when True (default) and a visual retriever is wired,
    # /answer dispatches via RoutingRetriever (text-only vs RRF-fused
    # text+visual per query category). False forces text-only — useful for
    # baseline comparisons or when the visual model is unavailable.
    enable_routing: bool = True
    # When True, `_wire_retriever_from_settings` attempts to build the visual
    # leg (ColQwen2 multivector index) and the LLM classifier so /answer
    # serves the full multi-modal stack. Default False because the visual
    # leg requires GPU + pre-rendered page PNGs (`pages_dir`) and 5-30 min
    # of startup time for embedding. When prerequisites aren't met, wiring
    # silently falls back to text-only — same behaviour as today's default
    # production deploy. Local dev with a populated `pages_dir` flips this
    # on to exercise the end-to-end multi-modal path.
    enable_multimodal: bool = False
    # Optional override for the LLM classifier model. When `enable_multimodal`
    # is on AND `openrouter_api_key` is set, RoutingRetriever uses the LLM
    # classifier (~$0.0001 per query) instead of ADR 0008's regex. The LLM
    # closes the dispatch gap on natural-language queries that don't carry
    # "Figure X" / "Table N" keywords (run cc45831697b6 → exp_classifier_dispatch:
    # regex 17 % vs LLM 72 % hybrid dispatch on MMLongBench).
    classifier_model: str = "openai/gpt-4o-mini"

    max_context_tokens: int = Field(default=8000, ge=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # When set, the production Generator attaches the rendered page PNG
    # (`<pages_dir>/<paper>/<paper>_pN.png`) for any visual RetrievalResult to
    # the LLM call as an OpenAI-compat content-block. Pair with a vision-capable
    # `default_chat_model` (gpt-4o-mini, gpt-4o, claude-sonnet-4.x, qwen3-vl,
    # …) — non-vision models will return 400 when sent images. Unset = text-only
    # behaviour (the previous default).
    pages_dir: Path | None = None

    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    # Collection wired into the FastAPI retriever at startup. Must match the
    # collection populated by `scripts/bootstrap_corpus.py` (its --collection
    # default is the same string). When the collection is missing or empty,
    # the API still boots — the retriever stays unwired and /answer returns
    # 503 until a corpus is ingested.
    corpus_collection: str = "rag_corpus"
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
