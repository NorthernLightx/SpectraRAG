"""Batch ingestion of recordings: document ids, collisions, failures."""

from __future__ import annotations

from pathlib import Path

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
