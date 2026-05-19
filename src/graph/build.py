"""Assemble a knowledge graph from chunk extractions and partition it (ADR 0018).

Entities merge by normalised name (case- and whitespace-insensitive) so the
same concept written two ways becomes one node; every node and edge keeps the
set of `chunk_ids` it came from — that provenance is what `GraphRetriever`
(S1.5) walks back to fetch source passages. Communities use networkx-native
Louvain (recursive for a 2-level hierarchy) rather than the hierarchical
Leiden of Microsoft GraphRAG: Leiden needs a heavy igraph/graspologic
dependency, and on a 20-paper demo corpus Louvain's partition is good enough.
ADR 0018 records the tradeoff.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from src.observability.logging import get_logger
from src.types import ChunkExtraction, Community

_log = get_logger(__name__)


def _norm(name: str) -> str:
    """Dedupe key: case- and whitespace-insensitive.

    Known lossy case: a case-only-distinct acronym pair (e.g. `mAP` the
    metric vs `MAP` the method) collapses into one node with an
    arbitrarily-tie-broken `type`. Accepted for the demo corpus pending the
    S1 spike measuring how often it actually occurs here; ADR 0018 records
    the decision and `test_graph_build` pins the behaviour so a future
    change is deliberate.
    """
    return " ".join(name.split()).lower()


def _touch_node(graph: nx.Graph, name: str, *, type_: str, description: str, chunk_id: str) -> str:
    key = _norm(name)
    if not key:
        return ""
    if key not in graph:
        graph.add_node(key, display=name, types=[], descriptions=[], chunk_ids=set())
    node = graph.nodes[key]
    node["types"].append(type_)
    if description and description not in node["descriptions"]:
        node["descriptions"].append(description)
    node["chunk_ids"].add(chunk_id)
    return key


def build_graph(extractions: list[ChunkExtraction]) -> nx.Graph:
    """Merge extractions into one undirected, weighted provenance graph."""
    graph: nx.Graph = nx.Graph()
    for ex in extractions:
        if ex.is_reference_list:
            continue  # bibliography / boilerplate — no graph signal (ADR 0017)
        for ent in ex.entities:
            _touch_node(
                graph, ent.name, type_=ent.type, description=ent.description, chunk_id=ex.chunk_id
            )
        for rel in ex.relations:
            src = _touch_node(
                graph, rel.source, type_="concept", description="", chunk_id=ex.chunk_id
            )
            tgt = _touch_node(
                graph, rel.target, type_="concept", description="", chunk_id=ex.chunk_id
            )
            if not src or not tgt or src == tgt:
                continue
            if graph.has_edge(src, tgt):
                edge = graph.edges[src, tgt]
                edge["weight"] += rel.weight
                if rel.description and rel.description not in edge["descriptions"]:
                    edge["descriptions"].append(rel.description)
                edge["chunk_ids"].add(ex.chunk_id)
            else:
                graph.add_edge(
                    src,
                    tgt,
                    weight=rel.weight,
                    descriptions=[rel.description] if rel.description else [],
                    chunk_ids={ex.chunk_id},
                )

    # Finalise: collapse the type multiset to its mode, sort provenance for
    # stable serialisation (S1.4) and deterministic tests.
    for _, attrs in graph.nodes(data=True):
        types: list[str] = attrs.pop("types")
        attrs["type"] = Counter(types).most_common(1)[0][0] if types else "concept"
        attrs["chunk_ids"] = sorted(attrs["chunk_ids"])
    for _, _, attrs in graph.edges(data=True):
        attrs["chunk_ids"] = sorted(attrs["chunk_ids"])
    _log.info("graph.built", nodes=graph.number_of_nodes(), edges=graph.number_of_edges())
    return graph


def detect_communities(graph: nx.Graph, *, max_size: int = 12, seed: int = 42) -> list[Community]:
    """Louvain partition; communities larger than `max_size` split once more.

    Deterministic for a fixed `seed`. Empty graph → no communities. A level-0
    community is only split when the recursion actually finds substructure
    (>1 sub-community), so small clusters stay whole.
    """
    if graph.number_of_nodes() == 0:
        return []

    communities: list[Community] = []
    level0: list[set[str]] = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    for i, members in enumerate(level0):
        cid = f"L0_{i}"
        communities.append(
            Community(community_id=cid, level=0, entity_names=sorted(members), parent_id=None)
        )
        if len(members) <= max_size:
            continue
        sub: list[set[str]] = nx.community.louvain_communities(
            graph.subgraph(members), weight="weight", seed=seed
        )
        if len(sub) < 2:
            continue
        for j, sub_members in enumerate(sub):
            communities.append(
                Community(
                    community_id=f"{cid}_s{j}",
                    level=1,
                    entity_names=sorted(sub_members),
                    parent_id=cid,
                )
            )
    by_level: dict[int, int] = Counter(c.level for c in communities)
    _log.info("graph.communities", total=len(communities), by_level=dict(by_level))
    return communities
