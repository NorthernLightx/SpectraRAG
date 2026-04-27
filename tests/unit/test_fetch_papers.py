"""ArXiv paper fetching: URL building + atom parsing + PDF download (mocked)."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from scripts.fetch_papers import arxiv_query_url, download_pdf, parse_arxiv_atom


def test_arxiv_query_url_builds_search_term() -> None:
    url = arxiv_query_url(category="cs.LG", max_results=3)
    assert "search_query=cat:cs.LG" in url
    assert "max_results=3" in url


_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>A test paper</title>
    <summary>Abstract text.</summary>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" />
    <author><name>A. Author</name></author>
  </entry>
</feed>"""


def test_parse_arxiv_atom_yields_paper_metadata() -> None:
    papers = parse_arxiv_atom(_SAMPLE_ATOM)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.00001v1"
    assert papers[0].title == "A test paper"
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2401.00001v1"
    assert "A. Author" in papers[0].authors


@respx.mock
async def test_download_pdf_writes_bytes(tmp_path: Path) -> None:
    respx.get("http://arxiv.org/pdf/2401.00001v1").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 fake")
    )
    path = await download_pdf("http://arxiv.org/pdf/2401.00001v1", "2401.00001v1", tmp_path)
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")
