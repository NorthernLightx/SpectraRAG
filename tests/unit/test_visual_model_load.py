"""Guards the ColQwen2 checkpoint against a silent weight-binding regression.

transformers 5.17 renames the vision-language module path (`model.*` becomes
`language_model.*`), which leaves `language_model.embed_tokens.weight` and
`language_model.norm.weight` unbound in the ColQwen2 checkpoint. They are then
newly initialised on every load. Nothing raises: the model loads, runs, and
returns a correctly-shaped embedding, so every page vector in the visual index
is quietly wrong. ADR 0007 pins the working set; this test is what notices when
a dependency bump breaks it.

The check is self-relative rather than a golden constant, so it holds across
torch builds and hardware: two loads of one fixed checkpoint must agree exactly.
Re-initialised weights are redrawn per load and diverge.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.rag.retrievers.visual import load_visual_model

_MODEL = "vidore/colqwen2-v1.0"
# ColQwen2 is a LoRA adapter over this base; both must already be cached.
_REQUIRED_CACHE_DIRS = (
    "models--vidore--colqwen2-v1.0",
    "models--Qwen--Qwen2-VL-2B-Instruct",
)


def _hf_cache() -> Path:
    import huggingface_hub.constants as hf

    return Path(hf.HF_HUB_CACHE)


def _weights_cached() -> bool:
    """True when both checkpoints are already on disk.

    Skips rather than downloads: the base model is ~4 GB, which is fine on a
    dev box that has already pulled it and wrong to fetch on a CI runner.
    """
    cache = _hf_cache()
    return all((cache / name).is_dir() for name in _REQUIRED_CACHE_DIRS)


def _embed_once() -> torch.Tensor:
    model, processor = asyncio.run(load_visual_model(_MODEL, "cpu"))
    batch = processor.process_images([Image.new("RGB", (224, 224), "white")])
    with torch.no_grad():
        embedding: torch.Tensor = model(**batch)
    return embedding.float().clone()


@pytest.mark.slow
@pytest.mark.skipif(not _weights_cached(), reason="ColQwen2 weights not in the local HF cache")
def test_colqwen2_weights_bind_deterministically() -> None:
    """Two loads of one checkpoint must produce identical embeddings.

    A mismatch means some parameter was re-initialised instead of loaded, which
    corrupts the page index without raising. Read the transformers load report
    for MISSING keys, then hold the dependency back.
    """
    first = _embed_once()
    second = _embed_once()

    assert first.shape == second.shape
    assert torch.equal(first, second), (
        "ColQwen2 embeddings differ across two loads of the same checkpoint, so "
        "weights are being randomly initialised. Check the transformers version."
    )
