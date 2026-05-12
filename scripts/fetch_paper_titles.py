"""Fetch arxiv titles for every paper in the current `pages_dir` corpus.

Writes `data/paper_titles.json` mapping `paper_id` → title. The /papers
endpoint reads this file to surface human-readable names in the demo UI
(falls back to paper_id when a title isn't listed).

Run via: `uv run python -m scripts.fetch_paper_titles`.

One arxiv API call per paper; respects the 3-second rate limit from
arxiv.org/help/api/user-manual#section_arxiv_api_misc_considerations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

_ARXIV_BASE = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_REQUEST_DELAY_SEC = 3.1  # arxiv asks for ≥3 seconds between requests


async def fetch_title(client: httpx.AsyncClient, arxiv_id: str) -> str | None:
    response = await client.get(_ARXIV_BASE, params={"id_list": arxiv_id})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    entry = root.find(f"{_ATOM_NS}entry")
    if entry is None:
        return None
    title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
    # The atom feed wraps long titles across lines; collapse runs of whitespace.
    return re.sub(r"\s+", " ", title) if title else None


def list_paper_ids(pages_dir: Path) -> list[str]:
    """Mirror src/api/routes/papers.py: paper_ids are pages_dir subdirectories
    with at least one matching PNG. Filtered to arxiv-format IDs since that's
    what we can resolve against the arxiv API.
    """
    if not pages_dir.is_dir():
        return []
    ids: list[str] = []
    for subdir in sorted(pages_dir.iterdir()):
        if not subdir.is_dir():
            continue
        paper_id = subdir.name
        if not _ARXIV_ID_RE.match(paper_id):
            continue
        if any(subdir.glob(f"{paper_id}_p*.png")):
            ids.append(paper_id)
    return ids


async def fetch_all(paper_ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for paper_id in paper_ids:
            try:
                title = await fetch_title(client, paper_id)
            except httpx.HTTPError as exc:
                print(f"WARN: {paper_id}: {exc}")
                title = None
            if title:
                out[paper_id] = title
                print(f"  {paper_id}: {title}")
            else:
                print(f"  {paper_id}: <not found>")
            await asyncio.sleep(_REQUEST_DELAY_SEC)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=Path("data/curated_demo/pages"),
        help="Directory containing per-paper PNG folders (default: data/curated_demo/pages)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/paper_titles.json"),
        help="Output JSON path (default: data/paper_titles.json)",
    )
    args = parser.parse_args()

    paper_ids = list_paper_ids(args.pages_dir)
    if not paper_ids:
        print(f"No arxiv-format papers found under {args.pages_dir}")
        return

    print(
        f"Resolving {len(paper_ids)} titles from arxiv (≈{len(paper_ids) * _REQUEST_DELAY_SEC:.0f}s)..."
    )
    titles = asyncio.run(fetch_all(paper_ids))

    # Merge with any existing file so manually-edited entries survive a re-run.
    existing: dict[str, str] = {}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged = {**existing, **titles}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(merged)} titles to {args.out}")


if __name__ == "__main__":
    main()
