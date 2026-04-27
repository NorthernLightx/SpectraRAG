"""Fetch ArXiv ML papers as PDFs. Run via: `uv run python -m scripts.fetch_papers`."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

_ARXIV_BASE = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class ArxivPaper:
    """Minimal metadata from an ArXiv search result."""

    arxiv_id: str
    title: str
    summary: str
    pdf_url: str
    authors: list[str]


def arxiv_query_url(*, category: str, max_results: int) -> str:
    params = {
        "search_query": f"cat:{category}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{_ARXIV_BASE}?{urllib.parse.urlencode(params, safe=':')}"


def parse_arxiv_atom(atom_xml: str) -> list[ArxivPaper]:
    """Parse an ArXiv Atom feed into ArxivPaper records."""
    root = ET.fromstring(atom_xml)
    papers: list[ArxivPaper] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        raw_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
        summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()
        pdf_url = ""
        for link in entry.findall(f"{_ATOM_NS}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        authors = [
            (a.findtext(f"{_ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{_ATOM_NS}author")
        ]
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                summary=summary,
                pdf_url=pdf_url,
                authors=[a for a in authors if a],
            )
        )
    return papers


async def download_pdf(pdf_url: str, arxiv_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id)
    out_path = out_dir / f"{safe_name}.pdf"
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(pdf_url)
        response.raise_for_status()
        out_path.write_bytes(response.content)
    return out_path


async def main(*, category: str, max_results: int, out_dir: Path) -> None:
    url = arxiv_query_url(category=category, max_results=max_results)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    papers = parse_arxiv_atom(response.text)
    for paper in papers:
        if not paper.pdf_url:
            continue
        path = await download_pdf(paper.pdf_url, paper.arxiv_id, out_dir)
        print(f"Saved {paper.arxiv_id}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ArXiv ML papers as PDFs.")
    parser.add_argument("--category", default="cs.LG")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("data/papers"))
    args = parser.parse_args()
    try:
        asyncio.run(
            main(category=args.category, max_results=args.max_results, out_dir=args.out_dir)
        )
    except KeyboardInterrupt:
        sys.exit(130)
