"""Knowledge-graph types for the GraphRAG tier (ADR 0018).

One `ChunkExtraction` per clean chunk: the entities and relations an LLM
read out of it, plus `is_reference_list` — the bibliography filter ADR 0017
deferred to this pass (a reference-list chunk yields no real entities and is
flagged here instead of by a lexical heuristic that provably cannot do it).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Closed vocabulary keeps the graph clean: a small type set means duplicate
# concepts merge instead of scattering across free-form labels. The extractor
# prompt pins these; unknown types are coerced to "concept" at build time.
ENTITY_TYPES = ("concept", "method", "model", "dataset", "metric", "task", "finding")


class GraphEntity(BaseModel):
    """A node: a named thing the paper talks about."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    description: str


class GraphRelation(BaseModel):
    """A directed, weighted edge between two entity names."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    description: str
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class ChunkExtraction(BaseModel):
    """What the extractor read out of one chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    is_reference_list: bool = False
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)


class Community(BaseModel):
    """A detected cluster of entities. `level` 0 is the coarse partition;
    a level-1 community has a `parent_id` (the level-0 community it splits)."""

    model_config = ConfigDict(frozen=True)

    community_id: str
    level: int = Field(ge=0)
    entity_names: list[str]
    parent_id: str | None = None


class CommunityReport(BaseModel):
    """An LLM summary of one community — what GraphRAG global search reads
    instead of raw passages. Spike-scoped: title + summary only."""

    model_config = ConfigDict(frozen=True)

    community_id: str
    title: str
    summary: str
