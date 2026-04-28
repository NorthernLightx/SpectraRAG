"""Golden-set YAML loader: parses, validates, round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.golden_set import dump_golden_set, load_golden_set
from src.types import GoldenQuery, GoldenSet

_VALID_YAML = """\
name: phase1-text
version: v1
queries:
  - query_id: q1
    text: "What is the main contribution?"
    paper_id: 2604.22753v1
    category: factual
    relevant_chunk_ids:
      - "2604.22753v1::p2::c11"
    expected_facts:
      - "The paper introduces an inter-basin gain criterion."
  - query_id: q2
    text: "Will it rain on Mars tomorrow?"
    paper_id: 2604.22753v1
    category: out_of_corpus
"""


def test_load_golden_set_parses_valid_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "v1.yaml"
    yaml_path.write_text(_VALID_YAML, encoding="utf-8")

    gs = load_golden_set(yaml_path)

    assert isinstance(gs, GoldenSet)
    assert gs.name == "phase1-text"
    assert gs.version == "v1"
    assert len(gs.queries) == 2
    assert gs.queries[0].category == "factual"
    assert gs.queries[1].category == "out_of_corpus"


def test_load_golden_set_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_set(tmp_path / "missing.yaml")


def test_load_golden_set_rejects_non_mapping(tmp_path: Path) -> None:
    yaml_path = tmp_path / "list.yaml"
    yaml_path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_golden_set(yaml_path)


def test_load_golden_set_rejects_unknown_category(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        'name: x\nversion: v1\nqueries:\n  - query_id: q1\n    text: "hi"\n'
        "    paper_id: p1\n    category: not-a-real-category\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_golden_set(yaml_path)


def test_dump_then_load_round_trips(tmp_path: Path) -> None:
    original = GoldenSet(
        name="phase1",
        version="v1",
        queries=[
            GoldenQuery(
                query_id="q1",
                text="What is X?",
                paper_id="p1",
                category="factual",
                relevant_chunk_ids=["p1::p1::c0"],
                expected_facts=["X is the answer."],
                note="known answer",
            )
        ],
    )
    yaml_path = tmp_path / "out.yaml"
    dump_golden_set(original, yaml_path)
    restored = load_golden_set(yaml_path)
    assert restored == original
