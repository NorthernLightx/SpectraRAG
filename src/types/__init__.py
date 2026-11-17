"""Public types shared across modules."""

from src.types.documents import Bbox, Chunk, Figure, MediaSegment, Page, Paper, Table
from src.types.eval import (
    EvalRun,
    GenerationMetrics,
    GoldenQuery,
    GoldenSet,
    PerQueryResult,
    QueryCategory,
    RetrievalMetrics,
)
from src.types.generation import Answer, Citation, Context
from src.types.graph import (
    ChunkExtraction,
    Community,
    CommunityReport,
    GraphEntity,
    GraphRelation,
)
from src.types.retrieval import (
    Query,
    RankedChunk,
    RetrievalResponse,
    RetrievalResult,
    RetrievalSource,
    RoutingInfo,
)

__all__ = [
    "Answer",
    "Bbox",
    "Chunk",
    "ChunkExtraction",
    "Citation",
    "Community",
    "CommunityReport",
    "Context",
    "EvalRun",
    "Figure",
    "GenerationMetrics",
    "GoldenQuery",
    "GoldenSet",
    "GraphEntity",
    "GraphRelation",
    "MediaSegment",
    "Page",
    "Paper",
    "PerQueryResult",
    "Query",
    "QueryCategory",
    "RankedChunk",
    "RetrievalMetrics",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalSource",
    "RoutingInfo",
    "Table",
]
