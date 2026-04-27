"""Embedding protocol and concrete implementations."""

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.embeddings.protocol import Embedder

__all__ = ["Embedder", "OllamaBgeEmbedder"]
