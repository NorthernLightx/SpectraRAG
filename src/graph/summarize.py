"""LLM community reports — what GraphRAG global search reads instead of raw
passages (ADR 0018, spike-scope).

One report per community: flat, no hierarchical roll-up — that depth is
deferred until the kill-spike says GraphRAG is worth continuing. Same
posture as `graph_extract.py`: concurrency-bounded, per-community failure
degrades to "no report" rather than aborting the batch.
"""

from __future__ import annotations

import asyncio

import httpx
import networkx as nx

from src.llm.protocol import LLMClient, Message
from src.observability.logging import get_logger, timed_event
from src.prompts.loader import load_prompt_by_name
from src.types import Community, CommunityReport

_log = get_logger(__name__)

_CONTEXT_BUDGET = 3000  # cap so a large community can't blow the token budget


def _community_context(graph: nx.Graph, community: Community) -> str:
    members = [n for n in community.entity_names if n in graph]
    if not members:
        return ""
    lines: list[str] = []
    for name in members:
        node = graph.nodes[name]
        desc = node["descriptions"][0] if node.get("descriptions") else ""
        lines.append(f"- {node.get('display', name)} [{node.get('type', 'concept')}]: {desc}")
    sub = graph.subgraph(members)
    for src, tgt, attrs in sub.edges(data=True):
        rel = attrs["descriptions"][0] if attrs.get("descriptions") else "related to"
        lines.append(f"- {src} — {tgt}: {rel}")
    return "\n".join(lines)[:_CONTEXT_BUDGET]


def _parse(text: str, community_id: str) -> CommunityReport | None:
    body = text.strip()
    if not body:
        return None
    title, _, summary = body.partition("\n")
    return CommunityReport(
        community_id=community_id,
        title=" ".join(title.split())[:120] or "Untitled cluster",
        summary=" ".join(summary.split()),
    )


async def _one(
    graph: nx.Graph,
    community: Community,
    *,
    llm: LLMClient,
    model: str,
    system: str | None,
    user_template: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> CommunityReport | None:
    context = _community_context(graph, community)
    if not context:
        return None
    async with semaphore:
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user_template.format(context=context)))
        try:
            response = await llm.chat(
                messages=messages, model=model, temperature=0.0, max_tokens=max_tokens
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            _log.warning("graph_summary.failed", community=community.community_id, error=str(exc))
            return None
    return _parse(response.text, community.community_id)


async def summarize_communities(
    graph: nx.Graph,
    communities: list[Community],
    *,
    llm: LLMClient,
    model: str,
    concurrency: int = 4,
    max_tokens: int = 400,
) -> list[CommunityReport]:
    """One report per community. Communities with no usable context or whose
    LLM call fails are dropped (not emitted empty — global search must not
    read blank reports)."""
    if not communities:
        return []
    prompt = load_prompt_by_name("graph_community_summary")
    semaphore = asyncio.Semaphore(concurrency)
    with timed_event(
        _log, "graph_summary.done", n_communities=len(communities), model=model
    ) as ctx:
        results = await asyncio.gather(
            *(
                _one(
                    graph,
                    c,
                    llm=llm,
                    model=model,
                    system=prompt.system,
                    user_template=prompt.user_template,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
                for c in communities
            )
        )
        reports = [r for r in results if r is not None]
        ctx["reports"] = len(reports)
    return reports
