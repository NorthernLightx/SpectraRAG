"""Knowledge graph: build from chunk extractions, detect communities (ADR 0018)."""

from src.graph.build import build_graph, detect_communities

__all__ = ["build_graph", "detect_communities"]
