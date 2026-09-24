"""Which colpali-engine classes load a visual checkpoint, and with which
processor settings. Selection is by model id; nothing touches the network."""

from __future__ import annotations

from typing import Any

import pytest
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColQwen2Processor,
    ColQwen3_5,
    ColQwen3_5Processor,
)

import src.rag.retrievers.visual as visual


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("vidore/colqwen2-v1.0", (ColQwen2, ColQwen2Processor)),
        ("vidore/colqwen2.5-v0.2", (ColQwen2_5, ColQwen2_5_Processor)),
        ("athrael-soju/colqwen3.5-4.5B-v3", (ColQwen3_5, ColQwen3_5Processor)),
        ("vidore/colpali-v1.3", (ColPali, ColPaliProcessor)),
        # Declares ColQwen3_5 in its config.json; its id names no colpali family.
        ("vultr/VultronRetrieverFlash-Qwen3.5-0.8B", (ColQwen3_5, ColQwen3_5Processor)),
    ],
)
def test_model_id_selects_classes(model_name: str, expected: tuple[Any, Any]) -> None:
    assert visual._select_col_classes(model_name) == expected


def test_unknown_model_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unsupported visual model"):
        visual._select_col_classes("some-org/bert-base")


async def test_card_visual_token_budget_reaches_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vultron ships a 1280-token processor cap and is evaluated at 1792 (its
    model card); the loader passes the card's budget and leaves others alone."""
    seen: dict[str, Any] = {}

    class _Model:
        @classmethod
        def from_pretrained(cls, _name: str, **_kwargs: Any) -> _Model:
            return cls()

        def train(self, _mode: bool) -> None:
            return None

    class _Processor:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> _Processor:
            seen[name] = kwargs
            return cls()

    monkeypatch.setattr(visual, "_select_col_classes", lambda _name: (_Model, _Processor))

    vultron = "vultr/VultronRetrieverFlash-Qwen3.5-0.8B"
    await visual.load_visual_model(vultron, "cpu")
    await visual.load_visual_model("vidore/colqwen2-v1.0", "cpu")

    assert seen[vultron] == {"max_num_visual_tokens": 1792}
    assert seen["vidore/colqwen2-v1.0"] == {}
