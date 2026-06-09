"""ADR 0022 regression guard, corpus arm — catch a bad rebake of the real data.

The unit guard (`test_figure_caption_invariant.py`) pins the *code path*: the
view layer never surfaces a captioned picture as ``unlabeled``. This arm pins
the *committed data*: every figure/table chunk in the baked
``qdrant_local/rag_corpus`` (the snapshot the deploy serves and CI ships) is run
through the real `figures._to_browse_item`, and any chunk whose caption matches
the primary `Figure N` / `Table N` pattern must surface ``role != "unlabeled"``.

So a future ``--force`` re-bake that regresses the classifier — e.g. reverting
`table → figure`, or dropping the gallery's `unlabeled → figure` collapse — turns
CI red against the shipping corpus, not only against synthetic chunks.

Read-only: scrolling the store does not mutate the tracked sqlite (verified —
``git status`` stays clean after a scroll). The client is closed so the local
single-writer lock is released for any later test. If the snapshot is absent
(a lean checkout that never fetched it), the guard skips rather than fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.api.routes.figures import _to_browse_item
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk

# Same primary-caption contract as the unit arm (ADR 0022). Local on purpose.
_INVARIANT_CAPTION_RE = re.compile(
    r"^\s*(?:\d+\s+)?(?:figure|fig\.?|table|tab\.?)\s+[A-Z0-9]", re.IGNORECASE
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_DIR = _REPO_ROOT / "qdrant_local"
_COLLECTION = "rag_corpus"  # Settings.corpus_collection default
_EMBED_DIM = 1024  # bge-m3; must match the baked vectors for the client to open


async def _load_figure_chunks() -> list[Chunk]:
    store = QdrantVectorStore(
        url=f"path:{_CORPUS_DIR}", collection_name=_COLLECTION, dim=_EMBED_DIM
    )
    try:
        chunks = await store.scroll_chunks()
    finally:
        # Release qdrant-client local mode's portalocker write lock so a later
        # test opening the same path doesn't hit "already accessed by another
        # instance" (mirrors test_vectorstore.py's reopen idiom).
        await store._client.close()
    return [c for c in chunks if c.metadata.get("kind") in {"figure", "table"}]


async def test_committed_corpus_has_no_captioned_unlabeled_figures() -> None:
    if not (_CORPUS_DIR / "collection" / _COLLECTION).exists():
        pytest.skip(f"no baked corpus at {_CORPUS_DIR}/collection/{_COLLECTION}")

    fig_chunks = await _load_figure_chunks()
    assert fig_chunks, (
        "corpus loaded but has no figure/table chunks — wrong collection or bad bake?"
    )

    offenders: list[str] = []
    for chunk in fig_chunks:
        caption = chunk.text or ""
        if not _INVARIANT_CAPTION_RE.match(caption):
            continue
        item = _to_browse_item(chunk)
        if item is not None and item.role == "unlabeled":
            offenders.append(f"{chunk.chunk_id}: {' '.join(caption.split())[:70]!r}")

    assert not offenders, (
        "captioned figures surfaced as role=unlabeled in the baked corpus "
        "(ADR 0022 invariant breach — a rebake hid real figures):\n  " + "\n  ".join(offenders)
    )
