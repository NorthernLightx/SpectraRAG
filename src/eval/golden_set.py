"""Load and validate golden-set YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.types import GoldenSet


def load_golden_set(path: Path) -> GoldenSet:
    """Read a golden-set YAML and parse into a validated GoldenSet."""
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Golden set YAML must be a mapping at top level: {path}")
    return GoldenSet.model_validate(data)


def dump_golden_set(golden_set: GoldenSet, path: Path) -> None:
    """Round-trip a GoldenSet back to YAML for editing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(golden_set.model_dump(mode="json"), fh, sort_keys=False)
