"""Load versioned YAML prompt templates. Version is `{declared}-{content-hash}`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LIBRARY = Path(__file__).parent / "library"


@dataclass(frozen=True)
class Prompt:
    """A versioned prompt template. `user_template` uses str.format() placeholders."""

    name: str
    version: str
    system: str | None
    user_template: str

    def render(self, /, **kwargs: object) -> tuple[str | None, str]:
        """Return (system, rendered_user)."""
        return self.system, self.user_template.format(**kwargs)


def _content_hash(data: dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=64)
def load_prompt(path: Path) -> Prompt:
    """Read a prompt YAML and return a Prompt with auto-versioned hash suffix."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must be a mapping: {path}")
    if "name" not in data or "user_template" not in data:
        raise ValueError(f"Prompt YAML missing required keys 'name'/'user_template': {path}")

    declared_version = str(data.get("version", "v0"))
    return Prompt(
        name=str(data["name"]),
        version=f"{declared_version}-{_content_hash(data)}",
        system=str(data["system"]) if data.get("system") is not None else None,
        user_template=str(data["user_template"]),
    )


def load_prompt_by_name(name: str, *, library_dir: Path | None = None) -> Prompt:
    """Convenience: load `<library>/<name>.yaml`."""
    library = library_dir or DEFAULT_LIBRARY
    return load_prompt(library / f"{name}.yaml")
