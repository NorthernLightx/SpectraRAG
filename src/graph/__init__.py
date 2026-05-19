"""Knowledge graph: build from chunk extractions, detect & summarize communities (ADR 0018)."""

from src.graph.build import build_graph, detect_communities
from src.graph.summarize import summarize_communities

__all__ = ["build_graph", "detect_communities", "summarize_communities"]
