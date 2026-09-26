"""Recorded talks -> segment pages, keyframes and transcript chunks.

The segmentation rules are tested on synthetic 1 fps thumbnails, the transcript
rules on synthetic words; one test decodes a tiny generated video end to end.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.ingestion import media
from src.ingestion.media import (
    MediaManifest,
    SegmentationParams,
    Word,
    load_manifest,
    segment_frames,
    transcript_chunks,
    write_manifest,
)
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import MediaSegment
from tests.fakes import FakeEmbedder

PARAMS = SegmentationParams()


def _thumb(value: float) -> np.ndarray:
    return np.full((36, 64), value, dtype=np.float32)


def _frames(values: list[float]) -> list[tuple[float, np.ndarray, int]]:
    """One thumbnail per second; the payload is the frame's second."""
    return [(float(t), _thumb(v), t) for t, v in enumerate(values)]


def test_slide_changes_cut_segments_and_keep_the_last_frame() -> None:
    frames = _frames([0.0] * 10 + [100.0] * 10 + [200.0] * 10)
    out = segment_frames(frames, duration_s=30.0, params=PARAMS)
    assert [s for s, _ in out] == [
        MediaSegment(page=1, start_s=0.0, end_s=10.0),
        MediaSegment(page=2, start_s=10.0, end_s=20.0),
        MediaSegment(page=3, start_s=20.0, end_s=30.0),
    ]
    assert [k for _, k in out] == [9, 19, 29]


def test_keyframe_skips_a_fade_at_the_end_of_a_segment() -> None:
    # Second 10 is mid-fade: too close to slide A to cut, too far to be stable.
    frames = _frames([0.0] * 10 + [3.0] + [100.0] * 10)
    out = segment_frames(frames, duration_s=21.0, params=PARAMS)
    assert [(s.start_s, s.end_s) for s, _ in out] == [(0.0, 11.0), (11.0, 21.0)]
    assert out[0][1] == 9


def test_a_fade_frame_starting_a_segment_does_not_cause_a_second_cut() -> None:
    # Second 10 is mid-fade and different enough to cut. The new slide must be
    # compared with its own stable frames, not the ghost that opened it.
    frames = _frames([0.0] * 10 + [50.0] + [100.0] * 20)
    out = segment_frames(frames, duration_s=31.0, params=PARAMS)
    assert [(s.start_s, s.end_s) for s, _ in out] == [(0.0, 10.0), (10.0, 31.0)]
    assert out[1][1] == 30


def test_a_new_slide_on_the_same_template_still_cuts() -> None:
    # New text on an unchanged background moves a 64x36 grey thumbnail only a
    # few levels. Missing that cut would drop the first slide from the index.
    frames = _frames([0.0] * 10 + [6.0] * 10)
    out = segment_frames(frames, duration_s=20.0, params=PARAMS)
    assert [(s.start_s, s.end_s) for s, _ in out] == [(0.0, 10.0), (10.0, 20.0)]


def test_a_long_static_slide_is_split_at_the_cap() -> None:
    frames = _frames([0.0] * 200)
    out = segment_frames(frames, duration_s=200.0, params=PARAMS)
    assert [(s.start_s, s.end_s) for s, _ in out] == [(0.0, 90.0), (90.0, 180.0), (180.0, 200.0)]


def test_a_change_inside_the_minimum_length_does_not_cut() -> None:
    frames = _frames([0.0] * 3 + [100.0] * 10)
    out = segment_frames(frames, duration_s=13.0, params=PARAMS)
    assert [(s.start_s, s.end_s) for s, _ in out] == [(0.0, 13.0)]
    assert out[0][1] == 12


def test_no_frames_fails_loudly() -> None:
    with pytest.raises(ValueError):
        segment_frames([], duration_s=10.0, params=PARAMS)


SEGMENTS = [
    MediaSegment(page=1, start_s=0.0, end_s=10.0),
    MediaSegment(page=2, start_s=10.0, end_s=20.0),
    MediaSegment(page=3, start_s=20.0, end_s=30.0),
]


def test_words_go_to_the_segment_holding_their_midpoint() -> None:
    words = [
        Word(start_s=1.0, end_s=1.5, text=" Hello"),
        Word(start_s=9.6, end_s=10.2, text=" world."),  # midpoint 9.9 -> page 1
        Word(start_s=21.0, end_s=21.4, text=" Bye."),
    ]
    chunks = transcript_chunks("talk", SEGMENTS, words)
    assert [(c.chunk_id, c.page_numbers, c.text) for c in chunks] == [
        ("talk::p1::c0", [1], "Hello world."),
        ("talk::p3::c1", [3], "Bye."),
    ]
    assert chunks[0].metadata == {"kind": "transcript", "start_s": 1.0, "end_s": 10.2}


def test_a_long_segment_splits_into_chunks_under_the_target() -> None:
    words = [Word(start_s=10 + i * 0.1, end_s=10 + i * 0.1 + 0.05, text=" word") for i in range(60)]
    chunks = transcript_chunks("talk", SEGMENTS, words, target_chars=100)
    assert len(chunks) == 3
    assert all(len(c.text) <= 100 for c in chunks)
    assert [c.chunk_id for c in chunks] == ["talk::p2::c0", "talk::p2::c1", "talk::p2::c2"]
    assert chunks[1].metadata["start_s"] == pytest.approx(chunks[0].metadata["end_s"] + 0.05)


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = MediaManifest(
        doc_id="talk",
        source="talk.mp4",
        duration_s=30.0,
        transcriber="fake",
        segmentation=PARAMS.as_dict(),
        segments=SEGMENTS,
    )
    write_manifest(tmp_path, manifest)
    assert (tmp_path / "talk" / "talk_media.json").exists()
    assert load_manifest(tmp_path, "talk") == manifest


class _FakeTranscriber:
    name = "fake"

    def __init__(self) -> None:
        self.ran_on: list[int] = []

    def transcribe(self, path: Path) -> list[Word]:
        self.ran_on.append(threading.get_ident())
        return [Word(start_s=2.0, end_s=2.5, text=" Pretraining"), Word(3.0, 3.4, " data.")]


async def test_ingest_media_indexes_chunks_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion.pipeline import ingest_media

    decoded_on: list[int] = []

    def fake_frames(
        path: Path, params: SegmentationParams
    ) -> tuple[Iterator[tuple[float, np.ndarray, Image.Image]], float]:
        decoded_on.append(threading.get_ident())
        red, blue = Image.new("RGB", (32, 18), "red"), Image.new("RGB", (32, 18), "blue")
        frames = [(float(t), _thumb(0.0), red) for t in range(10)]
        frames += [(float(t), _thumb(100.0), blue) for t in range(10, 20)]
        return iter(frames), 20.0

    monkeypatch.setattr(media, "iter_video_frames", fake_frames)
    transcriber = _FakeTranscriber()
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    result = await ingest_media(
        doc_id="talk",
        media_path=tmp_path / "talk.mp4",
        embedder=FakeEmbedder(dim=8),
        vectorstore=vectorstore,
        bm25=bm25,
        transcriber=transcriber,
        pages_dir=tmp_path / "pages",
    )

    main = threading.get_ident()
    assert decoded_on and decoded_on[0] != main
    assert transcriber.ran_on and transcriber.ran_on[0] != main
    assert [c.chunk_id for c in result.chunks] == ["talk::p1::c0"]
    assert result.chunks[0].text == "Pretraining data."
    assert (tmp_path / "pages" / "talk" / "talk_p1.png").exists()
    assert (tmp_path / "pages" / "talk" / "talk_p2.png").exists()
    manifest = load_manifest(tmp_path / "pages", "talk")
    assert [s.page for s in manifest.segments] == [1, 2]
    assert manifest.transcriber == "fake"
    assert [r.chunk_id for r in bm25.search("pretraining", top_k=5)] == ["talk::p1::c0"]


def test_segment_video_decodes_a_real_file(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "slides.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for color in ((255, 0, 0), (0, 0, 255), (0, 255, 0)):
            image = Image.new("RGB", (64, 48), color)
            for _ in range(5 * 8):  # 8 seconds per slide
                frame = av.VideoFrame.from_image(image)
                container.mux(stream.encode(frame))
        container.mux(stream.encode(None))

    segments = media.segment_video(path, "slides", tmp_path / "pages", PARAMS)
    assert [s.page for s in segments] == [1, 2, 3]
    assert segments[1].start_s == pytest.approx(8.0, abs=1.0)
    for page in (1, 2, 3):
        assert (tmp_path / "pages" / "slides" / f"slides_p{page}.png").exists()


class _FailingTranscriber:
    name = "fake"

    def transcribe(self, path: Path) -> list[Word]:
        raise AssertionError("transcribed again despite a cached transcript")


async def test_reingest_reuses_the_cached_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-segmenting must not pay for transcription again; the words only
    # depend on the audio and the transcriber.
    from src.ingestion.pipeline import ingest_media

    def fake_frames(
        path: Path, params: SegmentationParams
    ) -> tuple[Iterator[tuple[float, np.ndarray, Image.Image]], float]:
        image = Image.new("RGB", (32, 18), "red")
        return iter([(float(t), _thumb(0.0), image) for t in range(10)]), 10.0

    monkeypatch.setattr(media, "iter_video_frames", fake_frames)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await vectorstore.ensure_collection()
    runs = []
    for transcriber in (_FakeTranscriber(), _FailingTranscriber()):
        runs.append(
            await ingest_media(
                doc_id="talk",
                media_path=tmp_path / "talk.mp4",
                embedder=FakeEmbedder(dim=8),
                vectorstore=vectorstore,
                bm25=Bm25Index(),
                transcriber=transcriber,
                pages_dir=tmp_path / "pages",
            )
        )
    first, again = runs
    assert [c.text for c in again.chunks] == [c.text for c in first.chunks] == ["Pretraining data."]


def test_resegmenting_removes_keyframes_of_pages_that_no_longer_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The page index reads every keyframe on disk, so a leftover page 9 from an
    # earlier segmentation would be indexed as a real page.
    stale = tmp_path / "pages" / "talk" / "talk_p9.png"
    stale.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(stale)

    def fake_frames(
        path: Path, params: SegmentationParams
    ) -> tuple[Iterator[tuple[float, np.ndarray, Image.Image]], float]:
        image = Image.new("RGB", (32, 18), "red")
        return iter([(float(t), _thumb(0.0), image) for t in range(10)]), 10.0

    monkeypatch.setattr(media, "iter_video_frames", fake_frames)
    segments = media.segment_video(tmp_path / "talk.mp4", "talk", tmp_path / "pages", PARAMS)
    assert [s.page for s in segments] == [1]
    assert sorted(p.name for p in stale.parent.glob("*.png")) == ["talk_p1.png"]
