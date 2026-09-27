"""Recorded talks as documents whose pages are time segments.

A talk video becomes the page-shaped evidence the rest of the stack reads:

- Slide changes split it into segments. Segment N is page N of the document.
- Each segment's last stable frame is its page image,
  `<pages_dir>/<doc>/<doc>_p<N>.png`, so the visual index and the reader pick it
  up like a rendered PDF page.
- The speech inside a segment becomes that page's transcript chunks, which
  carry `start_s` and `end_s`.
- A manifest beside the keyframes, `<doc>_media.json`, keeps the segment times,
  so time-span labels map onto pages (src/eval/spans.py).

PyAV and faster-whisper come from the `media` extra and load only when a video
is decoded or transcribed.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel

from src.types import Chunk, MediaSegment

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image


@dataclass(frozen=True)
class SegmentationParams:
    """Slide-change detection on grey thumbnails sampled once per second.

    Differences are mean absolute grey levels (0-255) between thumbnails. A
    screen-shared slide deck is near 0 between frames of one slide and far above
    `cut_threshold` across a change; a webcam inset barely moves the mean.
    """

    sample_fps: float = 1.0
    thumb_width: int = 64
    thumb_height: int = 36
    # Distance from the segment's last stable frame that counts as a new slide.
    # New text on the same template moves the thumbnail only a few levels, and a
    # missed change drops a slide from the page index while a spurious cut only
    # splits one, so the threshold sits just above the noise.
    cut_threshold: float = 4.0
    # A frame this close to the one before it is stable, not mid-transition.
    stable_eps: float = 1.0
    min_segment_s: float = 5.0
    # Long static stretches still split, so one segment never holds minutes of speech.
    max_segment_s: float = 90.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _diff(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.abs(a - b).mean())


def segment_frames[T](
    frames: Iterable[tuple[float, NDArray[np.float32], T]],
    *,
    duration_s: float,
    params: SegmentationParams,
) -> list[tuple[MediaSegment, T]]:
    """Cut a stream of `(seconds, thumbnail, payload)` samples into segments,
    each with the payload of its keyframe.

    A frame starts a new segment when it differs from the current segment's
    last stable frame by more than `cut_threshold` and the segment is at least
    `min_segment_s` long, or when the segment reaches `max_segment_s`. The
    keyframe is the segment's last stable frame: at 1 fps a cross-fade lands a
    sample between two slides, and that ghost must not become the page image
    or the reference the next frames are compared with.
    """
    it = iter(frames)
    first = next(it, None)
    if first is None:
        raise ValueError("no video frames to segment")
    start_s, ref, key = first
    prev = ref
    out: list[tuple[float, float, T]] = []
    for t, thumb, payload in it:
        elapsed = t - start_s
        changed = _diff(thumb, ref) > params.cut_threshold
        if (changed and elapsed >= params.min_segment_s) or elapsed >= params.max_segment_s:
            out.append((start_s, t, key))
            start_s, ref, key, prev = t, thumb, payload, thumb
            continue
        if _diff(thumb, prev) <= params.stable_eps:
            ref, key = thumb, payload
        prev = thumb
    out.append((start_s, max(duration_s, start_s + 1e-3), key))
    return [
        (MediaSegment(page=n, start_s=a, end_s=b), k) for n, (a, b, k) in enumerate(out, start=1)
    ]


def iter_video_frames(
    path: Path, params: SegmentationParams
) -> tuple[Iterator[tuple[float, NDArray[np.float32], Image.Image]], float]:
    """Samples of `path` at `params.sample_fps` as (seconds, grey thumbnail,
    full RGB image), and the video's duration in seconds."""
    import av

    # PyAV's types are partial and absent without the media extra; keep them
    # out of the type check so every install checks the same way.
    container: Any = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if stream.duration is not None and stream.time_base is not None:
        duration_s = float(stream.duration * stream.time_base)
    else:
        duration_s = float(container.duration or 0) / 1_000_000
    step = 1.0 / params.sample_fps

    def samples() -> Iterator[tuple[float, NDArray[np.float32], Image.Image]]:
        next_t = 0.0
        try:
            for frame in container.decode(stream):
                if frame.time is None or frame.time < next_t:
                    continue
                next_t = frame.time + step
                thumb = frame.reformat(
                    width=params.thumb_width, height=params.thumb_height, format="gray"
                ).to_ndarray()
                yield float(frame.time), thumb.astype(np.float32), frame.to_image()
        finally:
            container.close()

    return samples(), duration_s


def keyframe_path(pages_dir: Path, doc_id: str, page: int) -> Path:
    return pages_dir / doc_id / f"{doc_id}_p{page}.png"


def segment_video(
    path: Path, doc_id: str, pages_dir: Path, params: SegmentationParams
) -> list[MediaSegment]:
    """Segment `path` and write each segment's keyframe as its page image,
    replacing the keyframes of any earlier segmentation."""
    frames, duration_s = iter_video_frames(path, params)
    segmented = segment_frames(frames, duration_s=duration_s, params=params)
    doc_dir = pages_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    # The page index reads every keyframe on disk; one left from a segmentation
    # with more pages would be indexed as a page that no longer exists.
    for old in doc_dir.glob(f"{glob.escape(doc_id)}_p*.png"):
        old.unlink()
    for segment, image in segmented:
        image.save(keyframe_path(pages_dir, doc_id, segment.page))
    return [segment for segment, _ in segmented]


@dataclass(frozen=True)
class Word:
    """A transcribed word. `text` keeps the transcriber's leading space."""

    start_s: float
    end_s: float
    text: str


class Transcriber(Protocol):
    name: str

    def transcribe(self, path: Path) -> list[Word]: ...


class WhisperTranscriber:
    """faster-whisper with word timestamps, on CPU in int8 by default. The
    model loads on the first call."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        threads: int = 8,
        language: str | None = None,
    ) -> None:
        # The name keys the transcript cache, so it names everything that
        # changes the words: None lets Whisper detect the language.
        self.name = f"faster-whisper/{model}/{compute_type}/{language or 'auto'}"
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._threads = threads
        self._language = language
        self._model: Any = None

    def transcribe(self, path: Path) -> list[Word]:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._threads,
            )
        segments, _ = self._model.transcribe(
            str(path), language=self._language, word_timestamps=True
        )
        return [
            Word(start_s=float(w.start), end_s=float(w.end), text=str(w.word))
            for segment in segments
            for w in segment.words or []
        ]


def transcript_chunks(
    doc_id: str,
    segments: Sequence[MediaSegment],
    words: Sequence[Word],
    *,
    target_chars: int = 1200,
) -> list[Chunk]:
    """Transcript chunks per segment page, at most `target_chars` each.

    A word belongs to the segment holding its midpoint. Chunks never cross a
    segment, so each one is evidence for exactly one page. Ids follow the text
    chunker's `{doc}::p{page}::c{counter}` with the counter running over the
    whole document.
    """
    by_page: dict[int, list[Word]] = {s.page: [] for s in segments}
    for word in words:
        mid = (word.start_s + word.end_s) / 2
        for segment in segments:
            if segment.start_s <= mid < segment.end_s:
                by_page[segment.page].append(word)
                break

    chunks: list[Chunk] = []

    def emit(page: int, group: list[Word]) -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::p{page}::c{len(chunks)}",
                paper_id=doc_id,
                page_numbers=[page],
                text="".join(w.text for w in group).strip(),
                metadata={
                    "kind": "transcript",
                    "start_s": group[0].start_s,
                    "end_s": group[-1].end_s,
                },
            )
        )

    for segment in segments:
        group: list[Word] = []
        for word in by_page[segment.page]:
            text = "".join(w.text for w in [*group, word]).strip()
            if group and len(text) > target_chars:
                emit(segment.page, group)
                group = []
            group.append(word)
        if group:
            emit(segment.page, group)
    return chunks


class MediaManifest(BaseModel):
    """Everything needed to map times onto a recording's pages and to tell
    whether a later ingest segmented or transcribed it differently."""

    doc_id: str
    source: str
    duration_s: float
    transcriber: str
    segmentation: dict[str, Any]
    segments: list[MediaSegment]
    # A video's pages have keyframes; an audio recording's pages are time
    # windows with transcript only.
    kind: Literal["video", "audio"] = "video"


def manifest_path(pages_dir: Path, doc_id: str) -> Path:
    return pages_dir / doc_id / f"{doc_id}_media.json"


def write_manifest(pages_dir: Path, manifest: MediaManifest) -> None:
    path = manifest_path(pages_dir, manifest.doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=1), encoding="utf-8")


def load_manifest(pages_dir: Path, doc_id: str) -> MediaManifest:
    return MediaManifest.model_validate_json(
        manifest_path(pages_dir, doc_id).read_text(encoding="utf-8")
    )


def words_path(pages_dir: Path, doc_id: str) -> Path:
    return pages_dir / doc_id / f"{doc_id}_words.json"


def media_fingerprint(path: Path) -> str:
    """Identity of a recording's bytes: its size and a hash of its first and
    last MiB. It survives a copy or a touch, and changes when the file is
    replaced by another take under the same name."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        digest.update(fh.read(1 << 20))
        fh.seek(max(0, size - (1 << 20)))
        digest.update(fh.read(1 << 20))
    return digest.hexdigest()[:16]


def save_words(
    pages_dir: Path, doc_id: str, transcriber: str, words: Sequence[Word], *, source: str
) -> None:
    path = words_path(pages_dir, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transcriber": transcriber,
        "source": source,
        "words": [[w.start_s, w.end_s, w.text] for w in words],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_words(pages_dir: Path, doc_id: str, transcriber: str, *, source: str) -> list[Word] | None:
    """The cached transcript of `doc_id`, or None unless it came from this
    transcriber and this recording (`media_fingerprint`). Segmentation never
    changes the words, so a re-segmentation reuses them."""
    path = words_path(pages_dir, doc_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("transcriber") != transcriber or payload.get("source") != source:
        return None
    return [Word(start_s=float(a), end_s=float(b), text=str(t)) for a, b, t in payload["words"]]


def cover_words(segments: Sequence[MediaSegment], words: Sequence[Word]) -> list[MediaSegment]:
    """`segments` with the last one stretched to the last word's end. A video's
    audio can outlast its last frame, and a word whose midpoint falls after
    every segment would otherwise belong to no page."""
    out = list(segments)
    last_end = max((w.end_s for w in words), default=0.0)
    if out and last_end >= out[-1].end_s:
        tail = out[-1]
        out[-1] = MediaSegment(page=tail.page, start_s=tail.start_s, end_s=last_end + 1e-3)
    return out


@dataclass(frozen=True)
class MediaProbe:
    has_video: bool
    has_audio: bool
    duration_s: float


def probe_media(path: Path) -> MediaProbe:
    """Which streams `path` holds, and its duration in seconds. Cover art in an
    audio file is a one-picture video stream, flagged as attached, and does not
    count as video."""
    import av

    container: Any = av.open(str(path))
    try:
        still = av.stream.Disposition.attached_pic | av.stream.Disposition.still_image
        has_video = any(not (s.disposition & still) for s in container.streams.video)
        if container.duration is not None:
            duration_s = container.duration / 1_000_000
        else:
            stream = container.streams[0]
            duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
        return MediaProbe(
            has_video=has_video,
            has_audio=len(container.streams.audio) > 0,
            duration_s=float(duration_s),
        )
    finally:
        container.close()


@dataclass(frozen=True)
class AudioSegmentationParams:
    """Time windows for a recording with no slides to cut on. A window closes at
    the first sentence end past `min_segment_s`, or at the first word past
    `max_segment_s` when no sentence ends, so a window never splits a sentence
    unless the speech runs on."""

    min_segment_s: float = 45.0
    max_segment_s: float = 90.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def segment_words(
    words: Sequence[Word], *, duration_s: float, params: AudioSegmentationParams
) -> list[MediaSegment]:
    """Contiguous windows over [0, duration_s) cut at word ends. Silence after
    the last word joins the last window instead of becoming an empty page."""
    cuts: list[float] = []
    start, has_words = 0.0, False
    for word in words:
        has_words = True
        elapsed = word.end_s - start
        sentence_end = word.text.rstrip().endswith((".", "?", "!"))
        if (sentence_end and elapsed >= params.min_segment_s) or elapsed >= params.max_segment_s:
            cuts.append(word.end_s)
            start, has_words = word.end_s, False
    if cuts and not has_words:
        cuts.pop()
    bounds = [0.0, *cuts, max(duration_s, cuts[-1] + 1e-3 if cuts else 1e-3)]
    return [
        MediaSegment(page=n, start_s=a, end_s=b)
        for n, (a, b) in enumerate(itertools.pairwise(bounds), start=1)
    ]
