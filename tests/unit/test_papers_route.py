"""GET /papers lists every document in the pages tree, recordings included."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.api.routes.papers import list_papers
from src.config.settings import Settings
from src.ingestion.media import MediaManifest, write_manifest
from src.types import MediaSegment


def test_papers_lists_page_images_and_audio_recordings(tmp_path: Path) -> None:
    (tmp_path / "doc").mkdir()
    for page in (1, 2):
        Image.new("RGB", (4, 4)).save(tmp_path / "doc" / f"doc_p{page}.png")
    # An audio recording has pages (time windows) but no page images.
    write_manifest(
        tmp_path,
        MediaManifest(
            doc_id="call",
            source="call.mp3",
            duration_s=120.0,
            transcriber="fake",
            segmentation={},
            segments=[
                MediaSegment(page=1, start_s=0.0, end_s=60.0),
                MediaSegment(page=2, start_s=60.0, end_s=120.0),
                MediaSegment(page=3, start_s=120.0, end_s=121.0),
            ],
            kind="audio",
        ),
    )
    papers = {p.paper_id: p.page_count for p in list_papers(settings=Settings(pages_dir=tmp_path))}
    assert papers == {"call": 3, "doc": 2}


def test_papers_tell_recordings_from_pdfs(tmp_path: Path) -> None:
    (tmp_path / "doc").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "doc" / "doc_p1.png")
    write_manifest(
        tmp_path,
        MediaManifest(
            doc_id="call",
            source="call.mp3",
            duration_s=121.0,
            transcriber="fake",
            segmentation={},
            segments=[MediaSegment(page=1, start_s=0.0, end_s=121.0)],
            kind="audio",
        ),
    )
    papers = {p.paper_id: p for p in list_papers(settings=Settings(pages_dir=tmp_path))}
    assert (papers["doc"].kind, papers["doc"].duration_s) == ("pdf", None)
    assert (papers["call"].kind, papers["call"].duration_s) == ("audio", 121.0)


def test_one_unreadable_manifest_does_not_take_down_the_list(tmp_path: Path) -> None:
    (tmp_path / "doc").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "doc" / "doc_p1.png")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "broken_media.json").write_text(
        '{"doc_id": "broken", "segm', encoding="utf-8"
    )
    papers = [p.paper_id for p in list_papers(settings=Settings(pages_dir=tmp_path))]
    assert papers == ["doc"]
