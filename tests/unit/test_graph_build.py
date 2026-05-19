"""Graph assembly + community detection (ADR 0018)."""

from __future__ import annotations

from src.graph import build_graph, detect_communities
from src.types import ChunkExtraction, GraphEntity, GraphRelation


def _ex(
    cid: str, ents: list[tuple[str, str]], rels: list[tuple[str, str, float]], *, ref: bool = False
) -> ChunkExtraction:
    return ChunkExtraction(
        chunk_id=cid,
        is_reference_list=ref,
        entities=[GraphEntity(name=n, type=t, description=f"{n} desc") for n, t in ents],
        relations=[
            GraphRelation(source=s, target=d, description=f"{s}->{d}", weight=w) for s, d, w in rels
        ],
    )


def test_entities_merge_by_normalized_name_with_provenance() -> None:
    g = build_graph(
        [
            _ex("c0", [("BGE-M3", "model")], [("BGE-M3", "Recall", 3.0)]),
            _ex("c1", [("bge-m3", "model"), ("Recall", "metric")], [("bge-m3", "Recall", 4.0)]),
        ]
    )
    assert "bge-m3" in g  # "BGE-M3" and "bge-m3" collapsed to one node
    assert sorted(g.nodes["bge-m3"]["chunk_ids"]) == ["c0", "c1"]
    assert g.edges["bge-m3", "recall"]["weight"] == 7.0  # summed across chunks
    assert sorted(g.edges["bge-m3", "recall"]["chunk_ids"]) == ["c0", "c1"]


def test_case_only_distinct_acronyms_collapse_known_loss() -> None:
    # Characterisation test (ADR 0018): `mAP` (metric) and `MAP` (method) are
    # distinct ML terms but the lowercase dedupe key merges them into one
    # node. Pinned so changing `_norm` is a deliberate, reviewed decision.
    g = build_graph(
        [
            _ex("c0", [("mAP", "metric")], []),
            _ex("c1", [("MAP", "method")], []),
        ]
    )
    assert "map" in g and g.number_of_nodes() == 1
    assert g.nodes["map"]["type"] in {"metric", "method"}  # tie broken arbitrarily


def test_reference_list_chunk_contributes_nothing() -> None:
    g = build_graph([_ex("c0", [("X", "concept")], [], ref=True)])
    assert g.number_of_nodes() == 0


def test_relation_endpoint_absent_from_entities_still_creates_node() -> None:
    g = build_graph([_ex("c0", [("A", "concept")], [("A", "B", 1.0)])])
    assert "b" in g and g.has_edge("a", "b")


def test_self_loops_skipped() -> None:
    g = build_graph([_ex("c0", [("A", "concept")], [("A", "a", 1.0)])])
    assert g.number_of_edges() == 0


def test_communities_split_large_and_are_deterministic() -> None:
    # Two dense clusters joined by a single bridge → ≥2 level-0 communities.
    ents = [(f"n{i}", "concept") for i in range(20)]
    rels = [(f"n{i}", f"n{i + 1}", 5.0) for i in range(9)]  # cluster A: n0..n9
    rels += [(f"n{i}", f"n{i + 1}", 5.0) for i in range(10, 19)]  # cluster B: n10..n19
    rels.append(("n9", "n10", 0.1))  # weak bridge
    g = build_graph([_ex("c0", ents, rels)])
    comms = detect_communities(g, seed=42)
    level0 = [c for c in comms if c.level == 0]
    assert len(level0) >= 2
    assert detect_communities(g, seed=42) == comms  # deterministic
    # Every level-1 community references a real level-0 parent.
    ids = {c.community_id for c in comms}
    assert all(c.parent_id in ids for c in comms if c.level == 1)


def test_communities_empty_graph() -> None:
    assert detect_communities(build_graph([])) == []
