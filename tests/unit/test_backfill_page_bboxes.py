"""backfill_page_bboxes: locating a stored chunk in a fresh parse, and the payload patch."""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from scripts.backfill_page_bboxes import Section, apply, find_blocks, squash
from src.ingestion.docling_chunker import _Block, section_text
from src.types import Bbox


def _block(text: str, page: int, y0: float = 100.0) -> _Block:
    return _Block(
        text=text, label="text", page=page, bbox=Bbox(x0=50.0, y0=y0, x1=300.0, y1=y0 + 40.0)
    )


def _section(title: str, *blocks: _Block) -> Section:
    body = list(blocks)
    letters, offsets = squash(section_text(body))
    return Section(title, body, letters, offsets)


def test_find_blocks_ignores_spacing_and_quote_style() -> None:
    fresh = _section("Results", _block('Accuracy drops by 0.30 on "MNIST" at most.', page=2))
    stored = "drops by 0 . 30 on 'MNIST' at"

    blocks = find_blocks(stored, [2], "Results", [fresh])

    assert blocks is not None and [b.page for b in blocks] == [2]


def test_find_blocks_returns_the_blocks_of_both_pages() -> None:
    fresh = _section(
        "Framework",
        _block("Before the chunk.", page=3, y0=600.0),
        _block("Definition of the term.", page=3, y0=700.0),
        _block("Role of exploration in training.", page=4, y0=80.0),
    )
    stored = "Definition of the term.\nRole of exploration"

    blocks = find_blocks(stored, [3, 4], "Framework", [fresh])

    assert blocks is not None and [b.text for b in blocks] == [
        "Definition of the term.",
        "Role of exploration in training.",
    ]


def test_find_blocks_rejects_text_on_other_pages() -> None:
    fresh = _section("Framework", _block("Definition of the term.", page=5))

    assert find_blocks("Definition of the term.", [3, 4], "Framework", [fresh]) is None
    assert find_blocks("Words the parse does not hold.", [5], "Framework", [fresh]) is None


def test_apply_adds_boxes_and_leaves_the_rest(tmp_path: Path) -> None:
    store = tmp_path / "qdrant"
    client = QdrantClient(path=str(store))
    client.create_collection(
        "rag_corpus", vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE)
    )
    payloads = [
        {
            "chunk_id": "p::p3::c1",
            "paper_id": "p",
            "page_numbers": [3, 4],
            "text": "kept",
            "metadata": {"section_rank": 2},
        },
        {
            "chunk_id": "p::p5::c2",
            "paper_id": "p",
            "page_numbers": [5],
            "text": "kept too",
            "metadata": {"bbox": [1.0, 2.0, 3.0, 4.0]},
        },
    ]
    client.upsert(
        "rag_corpus",
        points=[
            qm.PointStruct(id=i + 1, vector=[0.1 * (i + 1), 0.2], payload=payload)
            for i, payload in enumerate(payloads)
        ],
    )
    client.close()
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    per_page = {"3": [50.0, 700.0, 300.0, 740.0], "4": [50.0, 80.0, 300.0, 120.0]}
    (boxes / "p.json").write_text(json.dumps({"boxes": {"p::p3::c1": per_page}}), encoding="utf-8")

    assert apply(boxes, store, "rag_corpus") == 1
    assert apply(boxes, store, "rag_corpus") == 0  # a second run finds nothing to change

    client = QdrantClient(path=str(store))
    records, _ = client.scroll("rag_corpus", limit=10, with_payload=True, with_vectors=True)
    client.close()
    by_id = {str((r.payload or {})["chunk_id"]): r for r in records}
    patched, untouched = by_id["p::p3::c1"], by_id["p::p5::c2"]
    assert patched.payload == {
        **payloads[0],
        "metadata": {"section_rank": 2, "page_bboxes": per_page},
    }
    assert untouched.payload == payloads[1]
    assert patched.vector is not None and untouched.vector is not None
