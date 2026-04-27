"""Settings load order: defaults < YAML < env vars."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings, load_settings


def test_load_defaults_from_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text(
        "default_chat_model: anthropic/claude-3.5-sonnet\n"
        "default_embed_model: bge-m3\n"
        "top_k: 5\n"
        "rerank_top_k: 50\n"
    )
    settings = load_settings(config_path=yaml_file)
    assert settings.default_chat_model == "anthropic/claude-3.5-sonnet"
    assert settings.top_k == 5


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text("default_chat_model: from-yaml\n")
    monkeypatch.setenv("RAG_DEFAULT_CHAT_MODEL", "from-env")
    settings = load_settings(config_path=yaml_file)
    assert settings.default_chat_model == "from-env"


def test_missing_yaml_uses_defaults(tmp_path: Path) -> None:
    settings = load_settings(config_path=tmp_path / "does-not-exist.yaml")
    assert isinstance(settings, Settings)
    assert settings.top_k >= 1


def test_int_env_var_is_coerced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text("")
    monkeypatch.setenv("RAG_TOP_K", "12")
    settings = load_settings(config_path=yaml_file)
    assert settings.top_k == 12
