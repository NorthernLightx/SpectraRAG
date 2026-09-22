"""eval_run measures the retrieval stack it says it measures."""

from __future__ import annotations

import pytest

from scripts.eval_run import build_parser, retrieval_config_from_args
from src.config.settings import load_settings
from src.rag.retrieval_config import RetrievalConfig


def _config(*argv: str) -> RetrievalConfig:
    parser = build_parser()
    return retrieval_config_from_args(parser.parse_args(["--pdf", "x.pdf", *argv]), parser)


def test_profile_run_fingerprints_like_the_server_it_names() -> None:
    """The contract behind `--profile`: an eval of the cpu profile with the
    visual leg on and the API wired from that profile with RAG_ENABLE_MULTIMODAL
    report the same fingerprint (run JSON vs /health)."""
    served = load_settings(profile="cpu").model_copy(update={"enable_multimodal": True})
    assert _config("--profile", "cpu", "--router") == RetrievalConfig.from_settings(served)
    assert _config("--profile", "cpu").visual_model is None


def test_profile_rejects_explicit_retrieval_flags() -> None:
    with pytest.raises(SystemExit):
        _config("--profile", "cpu", "--rerank")


def test_force_route_hybrid_is_the_hybrid_mode() -> None:
    config = _config("--rerank", "--router", "--force-route", "hybrid")
    assert config.routing_mode == "hybrid"
    assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert config.classifier is None


def test_defaults_are_text_only_and_unreranked() -> None:
    config = _config()
    assert config.reranker_model is None
    assert config.visual_model is None
    assert config.embedder_backend == "ollama"
