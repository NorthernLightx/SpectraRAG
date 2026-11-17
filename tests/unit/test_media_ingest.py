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
from src.types import Chunk, MediaSegment
from tests.fakes import FakeEmbedder

PARAMS = SegmentationParams()


def _recording(path: Path) -> Path:
    """A stand-in file: probing and decoding are faked, the bytes only feed the
    transcript cache's fingerprint."""
    path.write_bytes(b"recording")
    return path


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
    monkeypatch.setattr(
        media,
        "probe_media",
        lambda path: media.MediaProbe(has_video=True, has_audio=True, duration_s=20.0),
    )
    transcriber = _FakeTranscriber()
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    result = await ingest_media(
        doc_id="talk",
        media_path=_recording(tmp_path / "talk.mp4"),
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

    probe = media.probe_media(path)
    assert (probe.has_video, probe.has_audio) == (True, False)
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
    monkeypatch.setattr(
        media,
        "probe_media",
        lambda path: media.MediaProbe(has_video=True, has_audio=True, duration_s=10.0),
    )
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await vectorstore.ensure_collection()
    runs = []
    for transcriber in (_FakeTranscriber(), _FailingTranscriber()):
        runs.append(
            await ingest_media(
                doc_id="talk",
                media_path=_recording(tmp_path / "talk.mp4"),
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


AUDIO = media.AudioSegmentationParams()


def _sentence(start: float, end: float, n: int = 5) -> list[Word]:
    """`n` words spread over [start, end), the last one ending a sentence."""
    step = (end - start) / n
    words = [Word(start + i * step, start + (i + 1) * step - 0.05, f" w{i}") for i in range(n)]
    last = words[-1]
    return [*words[:-1], Word(last.start_s, last.end_s, " end.")]


def test_audio_windows_cut_at_the_first_sentence_end_after_the_minimum() -> None:
    words = _sentence(0, 20) + _sentence(20, 50) + _sentence(50, 80) + _sentence(80, 100)
    segments = media.segment_words(words, duration_s=100.0, params=AUDIO)
    # 50 s is the first sentence end past 45 s; the rest never reaches 45 s again.
    assert [(s.page, s.start_s, round(s.end_s, 2)) for s in segments] == [
        (1, 0.0, 49.95),
        (2, 49.95, 100.0),
    ]


def test_audio_windows_force_a_cut_at_the_maximum_without_a_sentence_end() -> None:
    words = [Word(float(t), t + 0.5, " on") for t in range(0, 200)]
    segments = media.segment_words(words, duration_s=200.0, params=AUDIO)
    assert [s.end_s for s in segments][:2] == [90.5, 180.5]
    assert segments[-1].end_s == 200.0


def test_silence_after_the_last_word_joins_the_last_window() -> None:
    words = _sentence(0, 50) + _sentence(50, 70)
    segments = media.segment_words(words, duration_s=300.0, params=AUDIO)
    assert [s.page for s in segments] == [1, 2]
    assert segments[-1].end_s == 300.0


def test_a_recording_without_words_is_one_window() -> None:
    segments = media.segment_words([], duration_s=42.0, params=AUDIO)
    assert [(s.start_s, s.end_s) for s in segments] == [(0.0, 42.0)]


async def test_ingest_media_indexes_audio_without_keyframes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion.pipeline import ingest_media

    class _Talker:
        name = "fake"

        def transcribe(self, path: Path) -> list[Word]:
            return _sentence(0, 50) + _sentence(50, 100)

    monkeypatch.setattr(
        media,
        "probe_media",
        lambda path: media.MediaProbe(has_video=False, has_audio=True, duration_s=100.0),
    )
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await vectorstore.ensure_collection()
    result = await ingest_media(
        doc_id="call",
        media_path=_recording(tmp_path / "call.mp3"),
        embedder=FakeEmbedder(dim=8),
        vectorstore=vectorstore,
        bm25=Bm25Index(),
        transcriber=_Talker(),
        pages_dir=tmp_path / "pages",
    )
    manifest = load_manifest(tmp_path / "pages", "call")
    assert manifest.kind == "audio"
    assert [s.page for s in manifest.segments] == [1, 2]
    assert [c.page_numbers for c in result.chunks] == [[1], [2]]
    assert not list((tmp_path / "pages" / "call").glob("*.png"))


def test_probe_media_tells_audio_from_video(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "tone.wav"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=16000)
        samples = (np.sin(np.arange(16000 * 3) * 0.05) * 8000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    probe = media.probe_media(path)
    assert (probe.has_video, probe.has_audio) == (False, True)
    assert probe.duration_s == pytest.approx(3.0, abs=0.1)


def test_cover_art_does_not_make_an_audio_file_a_video(tmp_path: Path) -> None:
    # Podcasts and phone exports carry cover art as a one-picture video stream;
    # taking it for video would segment the recording as one slide.
    av = pytest.importorskip("av")
    path = tmp_path / "episode.mp4"
    with av.open(str(path), "w") as container:
        audio = container.add_stream("aac", rate=16000)
        cover = container.add_stream("mjpeg", rate=1)
        cover.width, cover.height, cover.pix_fmt = 64, 64, "yuvj420p"
        cover.disposition = av.stream.Disposition.attached_pic
        for packet in cover.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 64), "red"))):
            container.mux(packet)
        for packet in cover.encode(None):
            container.mux(packet)
        tone = (np.sin(np.arange(16000 * 2) * 0.05) * 0.2).astype(np.float32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(tone, format="fltp", layout="mono")
        frame.sample_rate = 16000
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)
    with av.open(str(path)) as check:
        assert len(check.streams.video) == 1
    assert media.probe_media(path).has_video is False


def test_words_after_the_last_segment_extend_it() -> None:
    # An audio track can run past the video's last frame; its last words must
    # not fall outside every page.
    segments = [MediaSegment(page=1, start_s=0.0, end_s=10.0)]
    words = [Word(9.0, 9.5, " late"), Word(10.2, 10.9, " words.")]
    covered = media.cover_words(segments, words)
    assert covered[-1].end_s >= 10.9
    chunks = transcript_chunks("talk", covered, words)
    assert chunks[0].text == "late words."


def test_cached_transcript_is_ignored_when_the_recording_changes(tmp_path: Path) -> None:
    recording = tmp_path / "talk.mp3"
    recording.write_bytes(b"first take" * 100)
    words = [Word(0.0, 0.5, " hello")]
    media.save_words(tmp_path, "talk", "fake", words, source=media.media_fingerprint(recording))
    assert media.load_words(tmp_path, "talk", "fake", source=media.media_fingerprint(recording))
    recording.write_bytes(b"second take" * 100)
    assert (
        media.load_words(tmp_path, "talk", "fake", source=media.media_fingerprint(recording))
        is None
    )


async def test_video_without_audio_is_indexed_without_transcribing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion.pipeline import ingest_media

    def fake_frames(
        path: Path, params: SegmentationParams
    ) -> tuple[Iterator[tuple[float, np.ndarray, Image.Image]], float]:
        image = Image.new("RGB", (32, 18), "red")
        return iter([(float(t), _thumb(0.0), image) for t in range(10)]), 10.0

    monkeypatch.setattr(media, "iter_video_frames", fake_frames)
    monkeypatch.setattr(
        media,
        "probe_media",
        lambda path: media.MediaProbe(has_video=True, has_audio=False, duration_s=10.0),
    )
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"x")
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await vectorstore.ensure_collection()
    result = await ingest_media(
        doc_id="silent",
        media_path=video,
        embedder=FakeEmbedder(dim=8),
        vectorstore=vectorstore,
        bm25=Bm25Index(),
        transcriber=_FailingTranscriber(),
        pages_dir=tmp_path / "pages",
    )
    assert result.chunk_count == 0
    assert (tmp_path / "pages" / "silent" / "silent_p1.png").exists()


def test_transcriber_cache_key_carries_the_language() -> None:
    assert media.WhisperTranscriber().name.endswith("/auto")
    assert media.WhisperTranscriber(language="en").name.endswith("/en")


def test_keyframes_reach_disk_while_the_video_is_still_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Holding every keyframe until the end costs gigabytes on long footage.
    pages = tmp_path / "pages"
    staged_midway: list[int] = []

    def frames() -> Iterator[tuple[float, np.ndarray, Image.Image]]:
        for t in range(40):
            if t == 31:  # frame 30 has closed the third slide
                staged_midway.append(len(list((pages / "talk").rglob("*.png"))))
            yield float(t), _thumb(100.0 * (t // 10)), Image.new("RGB", (8, 8), "red")

    monkeypatch.setattr(media, "iter_video_frames", lambda path, params: (frames(), 40.0))
    segments = media.segment_video(tmp_path / "talk.mp4", "talk", pages, PARAMS)
    assert [s.page for s in segments] == [1, 2, 3, 4]
    assert staged_midway == [3]
    assert sorted(p.name for p in (pages / "talk").iterdir()) == [
        f"talk_p{n}.png" for n in (1, 2, 3, 4)
    ]


def test_a_failed_decode_keeps_the_previous_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = tmp_path / "pages"
    (pages / "talk").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(pages / "talk" / "talk_p1.png")

    def broken() -> Iterator[tuple[float, np.ndarray, Image.Image]]:
        for t in range(25):
            yield float(t), _thumb(100.0 * (t // 10)), Image.new("RGB", (8, 8), "red")
        raise RuntimeError("corrupt stream")

    monkeypatch.setattr(media, "iter_video_frames", lambda path, params: (broken(), 40.0))
    with pytest.raises(RuntimeError):
        media.segment_video(tmp_path / "talk.mp4", "talk", pages, PARAMS)
    assert sorted(p.name for p in (pages / "talk").iterdir()) == ["talk_p1.png"]


def _audio_probe(path: Path) -> media.MediaProbe:
    return media.MediaProbe(has_video=False, has_audio=True, duration_s=100.0)


class _Speaker:
    name = "fake"

    def __init__(self, words: list[Word]) -> None:
        self.words = words

    def transcribe(self, path: Path) -> list[Word]:
        return self.words


class _BrokenStore(QdrantVectorStore):
    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise ConnectionError("qdrant went away")


async def test_the_manifest_is_written_only_once_the_chunks_are_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The batch script skips any recording with a manifest; one written before
    # a failed index would leave the recording unsearchable for good.
    from src.ingestion.pipeline import ingest_media

    monkeypatch.setattr(media, "probe_media", _audio_probe)
    store = _BrokenStore(url=":memory:", collection_name="media", dim=8)
    await store.ensure_collection()
    with pytest.raises(ConnectionError):
        await ingest_media(
            doc_id="call",
            media_path=_recording(tmp_path / "call.mp3"),
            embedder=FakeEmbedder(dim=8),
            vectorstore=store,
            bm25=Bm25Index(),
            transcriber=_Speaker(_sentence(0, 50)),
            pages_dir=tmp_path / "pages",
        )
    assert not media.manifest_path(tmp_path / "pages", "call").exists()


async def test_reingesting_a_recording_replaces_its_old_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion.pipeline import ingest_media

    monkeypatch.setattr(media, "probe_media", _audio_probe)
    store = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await store.ensure_collection()
    recording = _recording(tmp_path / "call.mp3")
    for words in (_sentence(0, 50) + _sentence(50, 100), _sentence(0, 50)):
        recording.write_bytes(str(len(words)).encode())  # a new take: new transcript
        await ingest_media(
            doc_id="call",
            media_path=recording,
            embedder=FakeEmbedder(dim=8),
            vectorstore=store,
            bm25=Bm25Index(),
            transcriber=_Speaker(words),
            pages_dir=tmp_path / "pages",
        )
    assert [c.chunk_id for c in await store.scroll_chunks()] == ["call::p1::c0"]


async def test_audio_without_speech_fails_and_keeps_the_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An audio recording's only pages are its transcript: with no words there
    # is nothing to search, and the manifest would mark it done anyway.
    from src.ingestion.pipeline import ingest_media

    monkeypatch.setattr(media, "probe_media", _audio_probe)
    store = QdrantVectorStore(url=":memory:", collection_name="media", dim=8)
    await store.ensure_collection()
    recording = _recording(tmp_path / "call.mp3")
    pages = tmp_path / "pages"
    for take, words in enumerate((_sentence(0, 50), [])):
        recording.write_bytes(f"take {take}".encode())
        run = ingest_media(
            doc_id="call",
            media_path=recording,
            embedder=FakeEmbedder(dim=8),
            vectorstore=store,
            bm25=Bm25Index(),
            transcriber=_Speaker(words),
            pages_dir=pages,
        )
        if words:
            await run
        else:
            with pytest.raises(ValueError, match="no speech"):
                await run
    assert [c.chunk_id for c in await store.scroll_chunks()] == ["call::p1::c0"]
    assert media.manifest_path(pages, "call").exists()


def test_a_truncated_transcript_cache_is_a_miss(tmp_path: Path) -> None:
    # A run killed mid-write must not make every later ingest of the recording fail.
    words = [Word(start_s=0.0, end_s=0.4, text=" Hello.")]
    media.save_words(tmp_path, "call", "fake", words, source="abc")
    assert media.load_words(tmp_path, "call", "fake", source="abc") == words
    assert [p.name for p in (tmp_path / "call").iterdir()] == ["call_words.json"]
    path = media.words_path(tmp_path, "call")
    path.write_text(path.read_text(encoding="utf-8")[:20], encoding="utf-8")
    assert media.load_words(tmp_path, "call", "fake", source="abc") is None


def _late_video(path: Path, *, with_audio: bool) -> Path:
    """8 s of red then 8 s of blue, timestamped from 3 s, with silent audio
    from 3 s when `with_audio`."""
    from fractions import Fraction

    av = pytest.importorskip("av")
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=5)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        if with_audio:
            audio = out.add_stream("aac", rate=16000, layout="mono")
            for i in range(16 * 16000 // 1024):
                samples = av.AudioFrame.from_ndarray(
                    np.zeros((1, 1024), np.float32), format="fltp", layout="mono"
                )
                samples.sample_rate = 16000
                samples.pts, samples.time_base = 3 * 16000 + i * 1024, Fraction(1, 16000)
                out.mux(audio.encode(samples))
            out.mux(audio.encode(None))
        for i, color in enumerate([(255, 0, 0)] * 40 + [(0, 0, 255)] * 40):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), color))
            frame.pts, frame.time_base = 15 + i, Fraction(1, 5)
            out.mux(video.encode(frame))
        out.mux(video.encode(None))
    return path


@pytest.mark.parametrize("with_audio", [False, True])
def test_a_video_timestamped_from_a_late_start_counts_from_zero(
    tmp_path: Path, with_audio: bool
) -> None:
    # Transcript word times count from the first audio sample, whatever the
    # container's timestamps say.
    path = _late_video(tmp_path / "late.mkv", with_audio=with_audio)
    segments = media.segment_video(path, "late", tmp_path / "pages", PARAMS)
    assert segments[0].start_s == 0.0
    assert segments[1].start_s == pytest.approx(8.0, abs=1.0)
    assert segments[-1].end_s == pytest.approx(16.0, abs=1.0)


def test_words_before_the_first_segment_join_it() -> None:
    # A video whose first frame comes after its audio starts.
    segments = [MediaSegment(page=1, start_s=3.0, end_s=10.0)]
    covered = media.cover_words(segments, [Word(0.5, 1.2, " Hello."), Word(4.0, 4.5, " Slides.")])
    assert covered[0].start_s == 0.0
    chunks = transcript_chunks("talk", covered, [Word(0.5, 1.2, " Hello.")])
    assert chunks[0].text == "Hello."
