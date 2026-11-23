"""Add per-page boxes to the multi-page text chunks already in a Qdrant corpus.

A text chunk that runs across pages carries `metadata['page_bboxes']` (ADR
0021 amendment). Chunks ingested before that carry no box. `locate` re-parses
one PDF, finds each stored chunk's text among the parsed blocks and records the
boxes of the blocks it covers. `apply` writes them into the stored payloads.
Chunk ids, text and vectors are not touched.

The fresh parse can differ from the one that built the corpus: Docling
versions space inline math differently and emit different quote characters.
Text is therefore compared by its letters and digits alone, and a chunk is
located only when that sequence is one run inside a section, covering exactly
its stored pages. A chunk the fresh parse splits or orders differently gets no
box.

Single-page chunks already carry a box, so `locate` also reports how many of
those stored boxes the fresh parse reproduces: the evidence that its boxes can
be trusted for the multi-page chunks.

One PDF per `locate` process, because Docling's memory grows across documents.

  uv run python -m scripts.backfill_page_bboxes locate \\
      --pdf data/papers/2604.28182v1.pdf --qdrant-path qdrant_local --out boxes
  uv run python -m scripts.backfill_page_bboxes apply \\
      --boxes boxes --qdrant-path qdrant_local
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.ingestion.docling_chunker import (
    _Block,
    body_sections,
    contributing_blocks,
    page_boxes,
    section_text,
)
from src.ingestion.docling_parser import convert_with_docling

_TEXT_CHUNK_ID = re.compile(r"::c\d+$")

# Two boxes mark the same region when they overlap at least this much, and
# still point at the same place on the page above the looser bound.
_SAME_REGION_IOU = 0.9
_SAME_PLACE_IOU = 0.5


class Section(NamedTuple):
    title: str | None
    body: list[_Block]
    letters: str  # `section_text(body)` reduced by `squash`
    offsets: list[int]  # position in `section_text(body)` of each kept character


def squash(text: str) -> tuple[str, list[int]]:
    """The letters and digits of `text` (NFKC), and where each one came from."""
    kept: list[str] = []
    offsets: list[int] = []
    for i, char in enumerate(text):
        for part in unicodedata.normalize("NFKC", char):
            if part.isalnum():
                kept.append(part)
                offsets.append(i)
    return "".join(kept), offsets


def parsed_sections(doc: Any) -> list[Section]:
    out: list[Section] = []
    for title, body in body_sections(doc):
        letters, offsets = squash(section_text(body))
        out.append(Section(title, body, letters, offsets))
    return out


def find_blocks(
    text: str, pages: list[int], section: str | None, sections: list[Section]
) -> list[_Block] | None:
    """The blocks under `text` when it occurs as one run in a section, or None.

    The run has to cover exactly `pages`. The section the chunk was stored
    under is tried first, so repeated text resolves to it.
    """
    wanted, _ = squash(text)
    if not wanted:
        return None
    for candidate in sorted(sections, key=lambda s: s.title != section):
        at = candidate.letters.find(wanted)
        while at >= 0:
            start = candidate.offsets[at]
            end = candidate.offsets[at + len(wanted) - 1] + 1
            blocks = contributing_blocks(candidate.body, start, end)
            if sorted({b.page for b in blocks}) == pages:
                return blocks
            at = candidate.letters.find(wanted, at + 1)
    return None


def _iou(a: list[float], b: list[float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, width) * max(0.0, height)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _boxes(blocks: list[_Block], pages: list[int]) -> dict[str, list[float]]:
    """A box per stored page the blocks cover, keyed like `page_bboxes`."""
    found = page_boxes(blocks)
    return {str(page): found[page].as_list() for page in pages if page in found}


def _scroll(
    client: QdrantClient, collection: str, paper_id: str | None = None
) -> list[tuple[Any, dict[str, Any]]]:
    """`(point id, payload)` for the collection's text chunks, optionally of one paper."""
    only = (
        qm.Filter(must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))])
        if paper_id is not None
        else None
    )
    out: list[tuple[Any, dict[str, Any]]] = []
    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection,
            scroll_filter=only,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = dict(record.payload or {})
            if _TEXT_CHUNK_ID.search(str(payload.get("chunk_id", ""))):
                out.append((record.id, payload))
        if offset is None:
            return out


def locate(pdf: Path, qdrant_path: Path, collection: str, out_dir: Path) -> dict[str, Any]:
    paper_id = pdf.stem
    client = QdrantClient(path=str(qdrant_path))
    try:
        stored = _scroll(client, collection, paper_id)
    finally:
        client.close()

    conversion = convert_with_docling(pdf)
    sections = parsed_sections(conversion.document)

    stats = dict.fromkeys(
        (
            "single_with_box",
            "single_found",
            "single_same_region",
            "single_same_place",
            "multi",
            "multi_found",
            "multi_boxed",
            "multi_boxed_on_every_page",
        ),
        0,
    )
    off_place: list[tuple[float, str]] = []
    boxes: dict[str, dict[str, list[float]]] = {}
    for _point_id, payload in stored:
        pages = [int(p) for p in payload["page_numbers"]]
        blocks = find_blocks(str(payload["text"]), pages, payload.get("section"), sections)
        if len(pages) == 1:
            stored_box = (payload.get("metadata") or {}).get("bbox")
            if stored_box is None:
                continue
            stats["single_with_box"] += 1
            fresh = _boxes(blocks, pages).get(str(pages[0])) if blocks is not None else None
            if fresh is None:
                continue
            stats["single_found"] += 1
            overlap = _iou(fresh, [float(v) for v in stored_box])
            stats["single_same_region"] += overlap >= _SAME_REGION_IOU
            stats["single_same_place"] += overlap >= _SAME_PLACE_IOU
            if overlap < _SAME_PLACE_IOU:
                off_place.append((round(overlap, 3), str(payload["chunk_id"])))
            continue
        stats["multi"] += 1
        if blocks is None:
            continue
        stats["multi_found"] += 1
        per_page = _boxes(blocks, pages)
        if per_page:
            boxes[str(payload["chunk_id"])] = per_page
            stats["multi_boxed"] += 1
            stats["multi_boxed_on_every_page"] += len(per_page) == len(pages)

    report: dict[str, Any] = {
        "paper_id": paper_id,
        "docling": version("docling"),
        "failed_pages": conversion.failed_pages,
        "stats": stats,
        "single_off_place": sorted(off_place),
        "boxes": boxes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{paper_id}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def apply(boxes_dir: Path, qdrant_path: Path, collection: str) -> int:
    """Write the located boxes into the stored payloads; returns how many changed."""
    located: dict[str, dict[str, list[float]]] = {}
    for path in sorted(boxes_dir.glob("*.json")):
        located.update(json.loads(path.read_text(encoding="utf-8"))["boxes"])

    client = QdrantClient(path=str(qdrant_path))
    changed = 0
    try:
        for point_id, payload in _scroll(client, collection):
            per_page = located.get(str(payload["chunk_id"]))
            metadata = dict(payload.get("metadata") or {})
            if per_page is None or metadata.get("page_bboxes") == per_page:
                continue
            pages = {str(p) for p in payload["page_numbers"]}
            if len(pages) < 2 or not set(per_page) <= pages:
                raise ValueError(f"{payload['chunk_id']}: boxes for pages it does not span")
            metadata["page_bboxes"] = per_page
            client.overwrite_payload(
                collection, payload={**payload, "metadata": metadata}, points=[point_id]
            )
            changed += 1
    finally:
        client.close()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("locate", "apply"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--qdrant-path", type=Path, required=True)
        cmd.add_argument("--collection", default="rag_corpus")
    sub.choices["locate"].add_argument("--pdf", type=Path, required=True)
    sub.choices["locate"].add_argument("--out", type=Path, required=True)
    sub.choices["apply"].add_argument("--boxes", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "locate":
        report = locate(args.pdf, args.qdrant_path, args.collection, args.out)
        print(json.dumps({k: v for k, v in report.items() if k != "boxes"}))
    else:
        print(f"payloads changed: {apply(args.boxes, args.qdrant_path, args.collection)}")


if __name__ == "__main__":
    main()
