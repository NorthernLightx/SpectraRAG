"""Prompt YAML loader: reads template, computes content-hashed version, caches."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompts.loader import Prompt, load_prompt, load_prompt_by_name


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_prompt_parses_minimal_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        "name: test\nversion: v1\nuser_template: 'Hello {name}.'\n",
    )

    prompt = load_prompt(yaml_path)

    assert isinstance(prompt, Prompt)
    assert prompt.name == "test"
    assert prompt.version.startswith("v1-")
    assert prompt.system is None
    system, rendered = prompt.render(name="World")
    assert system is None
    assert rendered == "Hello World."


def test_load_prompt_with_system_block(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        "name: t\nversion: v2\nsystem: 'You are helpful.'\nuser_template: '{query}'\n",
    )
    prompt = load_prompt(yaml_path)
    assert prompt.system == "You are helpful."
    system, rendered = prompt.render(query="hi")
    assert system == "You are helpful."
    assert rendered == "hi"


def test_version_changes_when_content_changes(tmp_path: Path) -> None:
    p1 = tmp_path / "a.yaml"
    p2 = tmp_path / "b.yaml"
    _write_yaml(p1, "name: x\nversion: v1\nuser_template: 'A'\n")
    _write_yaml(p2, "name: x\nversion: v1\nuser_template: 'B'\n")

    v1 = load_prompt(p1).version
    v2 = load_prompt(p2).version
    assert v1 != v2


def test_load_prompt_rejects_missing_required_keys(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(yaml_path, "name: only-name\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_prompt(yaml_path)


def test_load_prompt_by_name_finds_default_answer_prompt() -> None:
    """Bundled answer.yaml must load and render with placeholders."""
    prompt = load_prompt_by_name("answer")
    assert prompt.name == "answer"
    assert prompt.system is not None
    assert "research assistant" in prompt.system.lower()
    _, rendered = prompt.render(query="What is X?", context="Some context")
    assert "What is X?" in rendered
    assert "Some context" in rendered
