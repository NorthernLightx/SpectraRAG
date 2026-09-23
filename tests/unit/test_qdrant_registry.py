"""The committed Qdrant registry lists the visual collection (ADR 0028).

Embedded Qdrant loads only the collections `qdrant_local/meta.json` lists, so a
built visual index missing from it is invisible and `RAG_ENABLE_MULTIMODAL=true`
serves text only. `meta.visual.json` is the registry the Cloud Build overlay
installs; the two must match.
"""

from __future__ import annotations

import json
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parents[2] / "qdrant_local"


def _collections(name: str) -> dict[str, object]:
    registry: dict[str, dict[str, object]] = json.loads(
        (SNAPSHOT / name).read_text(encoding="utf-8")
    )
    return registry["collections"]


def test_meta_json_registers_the_collections_the_overlay_serves() -> None:
    assert _collections("meta.json") == _collections("meta.visual.json")
