"""Batch ingestion of recordings: document ids, collisions, failures."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from scripts.ingest_media import doc_id_for, ingest_all, plan


def test_doc_ids_are_sanitised_like_uploads() -> None:
    assert doc_id_for(Path("My Talk (final).mp4")) == "My_Talk__final_"
    assert doc_id_for(Path("ES2004a.wav")) == "ES2004a"
    # A dot-only stem would name pages_dir itself or its parent.
    assert doc_id_for(Path("...mp4")) == "recording"
    assert doc_id_for(Path("..mp4")) == "recording"


def test_recordings_that_share_a_doc_id_are_refused() -> None:
    # talk.mp4 and talk.mp3 would overwrite each other's pages and chunks.
    with pytest.raises(ValueError, match="talk"):
        plan([Path("a/talk.mp4"), Path("a/talk.mp3"), Path("a/other.wav")])


def test_plan_maps_each_recording_to_its_doc_id() -> None:
    assert plan([Path("a/x y.mp4"), Path("a/z.wav")]) == {
        "x_y": Path("a/x y.mp4"),
        "z": Path("a/z.wav"),
    }


async def test_one_failed_recording_does_not_stop_the_batch() -> None:
    done: list[str] = []

    async def ingest_one(doc_id: str, path: Path) -> None:
        if doc_id == "broken":
            raise RuntimeError("corrupt container")
        done.append(doc_id)

    failures = await ingest_all(
        {"first": Path("first.mp4"), "broken": Path("broken.mp4"), "last": Path("last.wav")},
        ingest_one,
    )
    assert done == ["first", "last"]
    assert failures == ["broken: RuntimeError: corrupt container"]


def test_a_missing_media_extra_is_named_before_any_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from scripts.ingest_media import missing_media_modules

    assert missing_media_modules() == []
    monkeypatch.setitem(sys.modules, "av", None)
    assert missing_media_modules() == ["av"]


def _ingested(pages: Path, doc_id: str, transcriber: str) -> None:
    from src.ingestion import media
    from src.types import MediaSegment

    media.write_manifest(
        pages,
        media.MediaManifest(
            doc_id=doc_id,
            source=f"{doc_id}.mp3",
            duration_s=60.0,
            transcriber=transcriber,
            segmentation=media.AudioSegmentationParams().as_dict(),
            segments=[MediaSegment(page=1, start_s=0.0, end_s=60.0)],
            kind="audio",
        ),
    )


def _run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fresh: bool, fails: set[str]
) -> tuple[int, list[str]]:
    """`main` over a.mp3 and b.mp3 with a fake embedder and a fake ingest."""
    import argparse
    import asyncio

    import scripts.ingest_media as script
    from tests.fakes import FakeEmbedder

    media_dir = tmp_path / "media"
    media_dir.mkdir(exist_ok=True)
    for name in ("a.mp3", "b.mp3"):
        (media_dir / name).write_bytes(b"audio")
    ingested: list[str] = []

    async def fake_ingest(*, doc_id: str, **kwargs: object) -> object:
        if doc_id in fails:
            raise RuntimeError("decoder crashed")
        ingested.append(doc_id)
        return mock.Mock(chunk_count=1, chunks=[mock.Mock(page_numbers=[1])])

    monkeypatch.setattr(script, "build_embedder", lambda config, ollama_url: FakeEmbedder(dim=8))
    monkeypatch.setattr(script, "ingest_media", fake_ingest)
    args = argparse.Namespace(
        media_dir=media_dir,
        pages_dir=tmp_path / "pages",
        qdrant=":memory:",
        collection="media",
        profile="cpu",
        ollama="http://localhost:11434",
        asr_model="large-v3-turbo",
        language="en",
        threads=1,
        only=None,
        fresh=fresh,
    )
    return asyncio.run(script.main(args)), ingested


CURRENT = "faster-whisper/large-v3-turbo/int8/en"


def test_an_interrupted_fresh_run_leaves_no_recording_marked_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --fresh drops every chunk; a manifest left behind would make the next run
    # skip a recording that is no longer indexed.
    from src.ingestion.media import manifest_path

    for doc_id in ("a", "b"):
        _ingested(tmp_path / "pages", doc_id, CURRENT)
    status, _ = _run_main(tmp_path, monkeypatch, fresh=True, fails={"a", "b"})
    assert status == 1
    assert not manifest_path(tmp_path / "pages", "a").exists()
    assert not manifest_path(tmp_path / "pages", "b").exists()


def test_a_recording_ingested_with_other_settings_is_ingested_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ingested(tmp_path / "pages", "a", CURRENT)
    _ingested(tmp_path / "pages", "b", "faster-whisper/large-v3-turbo/int8/auto")
    status, ingested = _run_main(tmp_path, monkeypatch, fresh=False, fails=set())
    assert (status, ingested) == (0, ["b"])
