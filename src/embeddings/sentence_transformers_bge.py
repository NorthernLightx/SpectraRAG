"""BGE-M3 embeddings via sentence-transformers (in-process, torch backend).

For the Cloud Run deploy where Ollama isn't available. Same 1024-dim output
as `OllamaBgeEmbedder` so the baked qdrant_local snapshot is interchangeable.

The model weights are pre-downloaded at Docker build time (see Dockerfile
RUN step that constructs `SentenceTransformer(...)` once); first request
after a cold start incurs only the model-load cost (~2-3 s on CPU), not a
network fetch.
"""

from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from src.observability.logging import get_logger

_BGE_M3_DIM = 1024
_log = get_logger(__name__)


class SentenceTransformersBgeEmbedder:
    """BGE-M3 via sentence-transformers (torch). Duck-typed against the Embedder protocol.

    Sync `model.encode(...)` is wrapped in `asyncio.to_thread` to satisfy the
    async protocol without blocking the event loop.
    """

    def __init__(self, *, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        self._model_name = model_name
        # Model load is eager. On a fresh container with weights already
        # cached (Dockerfile bake) this runs in ~2-3 s on CPU.
        self._model = SentenceTransformer(model_name, device=device)
        _log.info("sentence_transformers.loaded", model=model_name, dim=_BGE_M3_DIM, device=device)

    @property
    def dim(self) -> int:
        return _BGE_M3_DIM

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _encode() -> list[list[float]]:
            # `encode` returns numpy ndarray[N, D]; tolist() yields nested floats.
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            return [list(v) for v in vectors.tolist()]

        return await asyncio.to_thread(_encode)
