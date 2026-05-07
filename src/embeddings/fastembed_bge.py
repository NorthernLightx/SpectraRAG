"""BGE-M3 embeddings via fastembed (Qdrant's ONNX-quantized inference).

In-process embedding for the Cloud Run deploy where Ollama isn't available.
Same 1024-dim output as `OllamaBgeEmbedder` so the baked qdrant_local
snapshot is interchangeable.

The model is pre-downloaded at Docker build time (see Dockerfile RUN step
that constructs `TextEmbedding(...)` once); first request after a cold
start incurs only the model-load cost (~3-5 s), not a network fetch.
"""

from __future__ import annotations

import asyncio

from fastembed import TextEmbedding

from src.observability.logging import get_logger

_BGE_M3_DIM = 1024
_log = get_logger(__name__)


class FastEmbedBgeEmbedder:
    """BGE-M3 via fastembed (ONNX). Duck-typed against the Embedder protocol.

    Sync `model.embed(...)` is wrapped in `asyncio.to_thread` to satisfy the
    async protocol without blocking the event loop.
    """

    def __init__(self, *, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        # Model load is eager — happens here, not on first call. On a fresh
        # container with the weights already cached (Dockerfile bake) this
        # runs in ~3-5 s on CPU.
        self._model = TextEmbedding(model_name=model_name)
        _log.info("fastembed.loaded", model=model_name, dim=_BGE_M3_DIM)

    @property
    def dim(self) -> int:
        return _BGE_M3_DIM

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # fastembed's `.embed(...)` returns a generator of numpy arrays.
        # Materialise into Python floats off the event loop.
        return await asyncio.to_thread(lambda: [vec.tolist() for vec in self._model.embed(texts)])
