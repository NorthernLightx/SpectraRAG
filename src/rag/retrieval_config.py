"""The retrieval stack as one value, built into retrievers by the API and the eval.

`src/api/bootstrap.py` and `scripts/eval_run.py` both construct a
`RetrievalConfig` and pass it to the builders below; the API reports its
fingerprint on /health and the eval writes it into the run JSON. Equal
fingerprints mean identical retrieval (ADR 0014 amendment).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.routing import RoutingMode, RoutingRetriever

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.embeddings.protocol import Embedder
    from src.rag.bm25 import Bm25Index
    from src.rag.retrievers.classifier_llm import LLMQueryClassifier
    from src.rag.retrievers.protocol import Retriever
    from src.rag.vectorstore import QdrantVectorStore
    from src.types import Chunk


@dataclass(frozen=True)
class RetrievalConfig:
    """Every knob that changes which pages come back for a query.

    Deployment details that don't (device, URLs, paths) stay out, so a GPU eval
    and a CPU deploy of the same stack share a fingerprint. `__post_init__`
    clears fields another field switches off (no reranker means no length
    norm; no visual leg means no routing), so those don't split fingerprints.
    It does not catch every equivalence: a `fusion_depth` at or under the
    request's top_k retrieves like None but fingerprints differently.
    """

    embedder_backend: str = "ollama"
    embed_model: str = "bge-m3"
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    rerank_length_norm: bool = False
    rerank_length_threshold: int | None = 300
    rerank_length_penalty: float | None = 0.5
    candidate_pool: int = 50
    rerank_input_size: int | None = 50
    exclude_decoration: bool = True
    # None = text-only: no visual leg, so none of the routing fields apply.
    visual_model: str | None = None
    routing_mode: RoutingMode | None = None
    # "regex" or "llm:<model>"; only read in `category` mode.
    classifier: str | None = None
    cascade_threshold: float | None = None
    visual_fusion_weight: float = 1.0

    def __post_init__(self) -> None:
        normal: dict[str, Any] = {}
        if self.reranker_model is None:
            normal.update(rerank_length_norm=False, rerank_input_size=None)
        if not normal.get("rerank_length_norm", self.rerank_length_norm):
            normal.update(rerank_length_threshold=None, rerank_length_penalty=None)
        if self.visual_model is None:
            normal.update(
                routing_mode=None,
                classifier=None,
                cascade_threshold=None,
                visual_fusion_weight=1.0,
            )
        else:
            mode = self.routing_mode or "category"
            normal["routing_mode"] = mode
            if mode != "category":
                normal["classifier"] = None
            elif self.classifier is None:
                normal["classifier"] = "regex"
            if mode != "cascade":
                normal["cascade_threshold"] = None
        for key, value in normal.items():
            object.__setattr__(self, key, value)

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalConfig:
        classifier: str | None = None
        if settings.routing_mode == "category":
            model = (
                settings.classifier_model
                if settings.openrouter_api_key is not None
                else settings.classifier_ollama_model
            )
            classifier = f"llm:{model}"
        return cls(
            embedder_backend=settings.embedder_backend,
            embed_model=settings.default_embed_model,
            reranker_model=settings.reranker_model,
            rerank_length_norm=settings.rerank_length_norm,
            candidate_pool=settings.rerank_top_k,
            rerank_input_size=settings.rerank_input_size,
            exclude_decoration=settings.exclude_decoration_chunks,
            visual_model=settings.visual_model if settings.enable_multimodal else None,
            routing_mode=settings.routing_mode,
            classifier=classifier,
            cascade_threshold=settings.cascade_confidence_threshold,
            visual_fusion_weight=settings.visual_fusion_weight,
        )

    def text_only(self) -> RetrievalConfig:
        """The same stack with the visual leg absent: what runs when it fails to load."""
        return replace(self, visual_model=None)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def build_reranker(config: RetrievalConfig, *, device: str | None = None) -> BgeReranker | None:
    if config.reranker_model is None:
        return None
    kwargs: dict[str, Any] = {}
    if config.rerank_length_threshold is not None:
        kwargs["length_threshold"] = config.rerank_length_threshold
    if config.rerank_length_penalty is not None:
        kwargs["length_penalty"] = config.rerank_length_penalty
    return BgeReranker(
        model_name=config.reranker_model,
        device=device,
        length_norm=config.rerank_length_norm,
        **kwargs,
    )


def build_text_retriever(
    config: RetrievalConfig,
    *,
    embedder: Embedder,
    vectorstore: QdrantVectorStore,
    bm25: Bm25Index,
    chunks_by_id: dict[str, Chunk],
    reranker: BgeReranker | None = None,
    reranker_device: str | None = None,
) -> PipelineRetriever:
    """The text leg. Pass `reranker` to reuse an already-loaded model."""
    if reranker is None:
        reranker = build_reranker(config, device=reranker_device)
    return PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        candidate_pool=config.candidate_pool,
        reranker=reranker,
        rerank_input_size=config.rerank_input_size or config.candidate_pool,
        exclude_decoration=config.exclude_decoration,
    )


def build_routing_retriever(
    config: RetrievalConfig,
    *,
    text: Retriever,
    visual: Retriever,
    classifier: LLMQueryClassifier | None = None,
) -> RoutingRetriever:
    """Wrap a text leg (possibly already decorated by eval-only wrappers) with
    the visual leg per `config`. `cascade` without a threshold raises here."""
    if config.visual_model is None or config.routing_mode is None:
        raise ValueError("build_routing_retriever needs a config with a visual leg")
    return RoutingRetriever(
        text=text,
        visual=visual,
        classifier=classifier,
        mode=config.routing_mode,
        cascade_confidence_threshold=config.cascade_threshold,
        visual_fusion_weight=config.visual_fusion_weight,
    )
