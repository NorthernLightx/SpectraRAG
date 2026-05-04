"""Settings load order: defaults < YAML < env vars."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from src.config.settings import Settings, load_settings


def test_load_defaults_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """YAML values flow through when no env var is set. Uses sentinel values
    so a developer's local .env (loaded at module import) cannot mask the
    YAML path under test."""
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text(
        "default_chat_model: yaml-sentinel-model\n"
        "default_embed_model: bge-m3\n"
        "top_k: 5\n"
        "rerank_top_k: 50\n"
    )
    monkeypatch.delenv("RAG_DEFAULT_CHAT_MODEL", raising=False)
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    settings = load_settings(config_path=yaml_file)
    assert settings.default_chat_model == "yaml-sentinel-model"
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


def test_openrouter_api_key_loads_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text("")
    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "sk-or-v1-test")
    settings = load_settings(config_path=yaml_file)
    assert isinstance(settings.openrouter_api_key, SecretStr)
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-v1-test"


def test_openrouter_api_key_defaults_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text("")
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    settings = load_settings(config_path=yaml_file)
    assert settings.openrouter_api_key is None


def test_default_chat_model_is_current_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned to Claude Sonnet 4.x family. Bump explicitly when upgrading."""
    monkeypatch.delenv("RAG_DEFAULT_CHAT_MODEL", raising=False)
    settings = Settings()
    assert settings.default_chat_model.startswith("anthropic/claude-sonnet-4"), (
        f"expected sonnet-4.x, got {settings.default_chat_model!r}"
    )


def test_enable_routing_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR 0008 §"Decision": routing on by default when a visual leg is available."""
    monkeypatch.delenv("RAG_ENABLE_ROUTING", raising=False)
    assert Settings().enable_routing is True


def test_enable_routing_can_be_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_ROUTING", "false")
    assert Settings().enable_routing is False


def test_visual_model_defaults_to_colqwen2_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR 0008 caveat — pinned to the 4 GB-VRAM-fitting checkpoint by default."""
    monkeypatch.delenv("RAG_VISUAL_MODEL", raising=False)
    assert Settings().visual_model == "vidore/colqwen2-v1.0"


def test_public_api_key_defaults_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No key = no auth applied (development default)."""
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text("")
    monkeypatch.delenv("RAG_PUBLIC_API_KEY", raising=False)
    settings = load_settings(config_path=yaml_file)
    assert settings.public_api_key is None


def test_public_api_key_loads_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "default.yaml"
    yaml_file.write_text("")
    monkeypatch.setenv("RAG_PUBLIC_API_KEY", "shared-secret-xyz")
    settings = load_settings(config_path=yaml_file)
    assert isinstance(settings.public_api_key, SecretStr)
    assert settings.public_api_key.get_secret_value() == "shared-secret-xyz"
