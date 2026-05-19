"""LLM entity/relation extraction over clean chunks — the GraphRAG indexing
pass (ADR 0018) and the bibliography filter ADR 0017 deferred here.

One `chat` call per chunk. Small local models are unreliable JSON emitters,
so parsing is deliberately tolerant (strip fences, take the outer object,
skip malformed entries individually) and a per-chunk failure degrades to an
empty extraction rather than aborting a ~2000-call batch — same graceful
posture as `captioner.py` / `contextualize.py`.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.prompts.loader import load_prompt_by_name
from src.types import Chunk, ChunkExtraction, GraphEntity, GraphRelation
from src.types.graph import ENTITY_TYPES

_log = get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _entities(raw: Any) -> list[GraphEntity]:
    out: list[GraphEntity] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        etype = str(item.get("type", "concept")).strip().lower() or "concept"
        if etype not in ENTITY_TYPES:
            etype = "concept"
        try:
            out.append(
                GraphEntity(
                    name=name, type=etype, description=str(item.get("description", "")).strip()
                )
            )
        except ValidationError:
            continue
    return out


def _relations(raw: Any) -> list[GraphRelation]:
    out: list[GraphRelation] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target:
            continue
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        try:
            out.append(
                GraphRelation(
                    source=source,
                    target=target,
                    description=str(item.get("description", "")).strip(),
                    weight=min(max(weight, 0.0), 10.0),
                )
            )
        except ValidationError:
            continue
    return out


def _parse_extraction(text: str, chunk_id: str) -> ChunkExtraction:
    """Best-effort parse of one LLM response into a ChunkExtraction."""
    body = text.strip()
    fence = _FENCE_RE.search(body)
    if fence:
        body = fence.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        _log.warning("graph_extract.no_json", chunk_id=chunk_id)
        return ChunkExtraction(chunk_id=chunk_id)
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        _log.warning("graph_extract.bad_json", chunk_id=chunk_id, error=str(exc))
        return ChunkExtraction(chunk_id=chunk_id)
    if not isinstance(data, dict):
        return ChunkExtraction(chunk_id=chunk_id)
    return ChunkExtraction(
        chunk_id=chunk_id,
        is_reference_list=bool(data.get("is_reference_list", False)),
        entities=_entities(data.get("entities")),
        relations=_relations(data.get("relations")),
    )


async def _extract_one(
    chunk: Chunk,
    *,
    llm: LLMClient,
    model: str,
    system: str | None,
    user_template: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> ChunkExtraction:
    async with semaphore:
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user_template.format(chunk_text=chunk.text)))
        try:
            response = await llm.chat(
                messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            _log.warning("graph_extract.llm_failed", chunk_id=chunk.chunk_id, error=str(exc))
            return ChunkExtraction(chunk_id=chunk.chunk_id)
    return _parse_extraction(response.text, chunk.chunk_id)


async def extract_graph(
    chunks: list[Chunk],
    *,
    llm: LLMClient,
    model: str,
    concurrency: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> list[ChunkExtraction]:
    """Extract entities/relations from every chunk. One LLM call per chunk.

    Reference-list / boilerplate chunks come back with `is_reference_list`
    True and no entities — that is the bibliography filter ADR 0017 deferred
    to the LLM. Concurrency-bounded so a small-VRAM Ollama host is not
    thrashed; failures are per-chunk, never fatal.
    """
    if not chunks:
        return []
    prompt = load_prompt_by_name("graph_extract")
    semaphore = asyncio.Semaphore(concurrency)
    with timed_event(_log, "graph_extract.done", n_chunks=len(chunks), model=model) as ctx:
        results = await asyncio.gather(
            *(
                _extract_one(
                    c,
                    llm=llm,
                    model=model,
                    system=prompt.system,
                    user_template=prompt.user_template,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
                for c in chunks
            )
        )
        ctx["ref_list_chunks"] = sum(1 for r in results if r.is_reference_list)
        ctx["entities"] = sum(len(r.entities) for r in results)
        ctx["relations"] = sum(len(r.relations) for r in results)
    return list(results)
