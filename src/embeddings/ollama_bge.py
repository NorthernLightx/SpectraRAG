"""BGE-M3 embeddings served by Ollama via HTTP. No SDK to keep deps minimal."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_BGE_M3_DIM = 1024
_DEFAULT_TIMEOUT_SECONDS = 60.0


class OllamaBgeEmbedder:
    """BGE-M3 via Ollama. Duck-typed against the Embedder protocol.

    Ollama's /api/embeddings accepts one prompt at a time; we issue requests
    sequentially here and let batched callers parallelise if needed.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        model: str = "bge-m3",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = client

    @property
    def dim(self) -> int:
        return _BGE_M3_DIM

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.RemoteProtocolError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _embed_one(self, text: str, client: httpx.AsyncClient) -> list[float]:
        response = await client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError(f"Ollama returned no embedding for model {self._model}")
        return [float(x) for x in embedding]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is not None:
            return [await self._embed_one(t, self._client) for t in texts]
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return [await self._embed_one(t, client) for t in texts]
