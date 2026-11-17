"""Recover MCIF's human answer locations as time spans, without new labelling.

MCIF's annotators marked each answerable question with an Answer Start and
Answer End in the talk; the benchmark build (hlt-mt/mcif,
dataset_build/testset_generator.py) turns that range into the short-form clip:
the automatic segment, or run of segments joined together, that overlaps it.
Only the clip is published. Finding the clip inside its talk's audio, by
normalised cross-correlation of its first and last seconds, gives the location
back in seconds, widened to the segment edges. Unanswerable questions get a
random clip in that build and are skipped.

The output has the labelling page's export format, so it feeds
`build_mcif_golden --spans` unchanged.

Usage:
    uv run python -m scripts.locate_mcif_spans --out data/mcif/mcif-spans.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.fetch_mcif import download, qa_samples
from src.types.eval import TimeSpan

# Seconds matched at each end of a clip; long enough to be unique in a talk.
_PROBE_S = 3.0
# A copy cut from the talk correlates near 1; anything else sits far below.
_MIN_SCORE = 0.9
# A joined clip skips the pauses between its segments, so its span in the talk
# can exceed its own length by those pauses, never by a whole stretch of talk.
_MAX_GAP_S = 10.0
_SHORT_REFERENCES = "MCIF.short.en.ref.xml.gz"


def _best_offset(probe: NDArray[np.float32], talk: NDArray[np.float32]) -> tuple[int, float]:
    """Sample offset where `probe` best matches `talk`, and the normalised
    correlation there (1.0 for an exact copy at any gain)."""
    n, size = len(probe), len(talk)
    fft_len = 1 << int(np.ceil(np.log2(size + n)))
    corr = np.fft.irfft(np.fft.rfft(talk, fft_len) * np.conj(np.fft.rfft(probe, fft_len)), fft_len)[
        : size - n + 1
    ]
    energy = np.concatenate([[0.0], np.cumsum(talk.astype(np.float64) ** 2)])
    window = np.sqrt(np.maximum(energy[n:] - energy[:-n], 1e-12))
    score = corr / (window * np.linalg.norm(probe) + 1e-12)
    best = int(np.argmax(score))
    return best, float(score[best])


def locate_clip(clip: NDArray[np.float32], talk: NDArray[np.float32], rate: int) -> TimeSpan | None:
    """The span of `talk` that `clip` was cut from, first sample to last, or
    None when either end does not match or the ends contradict the clip: a span
    shorter than the clip, or longer by more than the pauses a join can skip."""
    n = min(len(clip), int(_PROBE_S * rate))
    head, head_score = _best_offset(clip[:n], talk)
    tail, tail_score = _best_offset(clip[-n:], talk)
    start, end = head / rate, (tail + n) / rate
    clip_s = len(clip) / rate
    if min(head_score, tail_score) < _MIN_SCORE:
        return None
    if not clip_s - 0.5 <= end - start <= clip_s + _MAX_GAP_S:
        return None
    return TimeSpan(start_s=start, end_s=end)


def _read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit mono")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return (pcm / np.float32(32768)).astype(np.float32), int(w.getframerate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mcif"))
    parser.add_argument("--out", type=Path, default=Path("data/mcif/mcif-spans.json"))
    args = parser.parse_args()

    with gzip.open(download("MCIF.long.en.ref.xml.gz"), "rt", encoding="utf-8") as fh:
        long_samples = {s.iid: s for s in qa_samples(fh.read())}
    with gzip.open(download(_SHORT_REFERENCES), "rt", encoding="utf-8") as fh:
        short_samples = {s.iid: s for s in qa_samples(fh.read())}

    spans: dict[str, list[list[float]]] = {}
    failed: list[str] = []
    widths: list[float] = []
    talks: dict[str, tuple[NDArray[np.float32], int]] = {}
    answerable = [iid for iid, s in long_samples.items() if s.qa_type != "NA"]
    for iid in answerable:
        talk_id, clip_id = long_samples[iid].talk, short_samples[iid].talk
        if talk_id not in talks:
            path = download(f"MCIF_DATA/LONG_AUDIOS/{talk_id}.wav", local_dir=args.data_dir)
            talks[talk_id] = _read_wav(path)
        talk, rate = talks[talk_id]
        clip, clip_rate = _read_wav(
            download(f"MCIF_DATA/SHORT_AUDIOS/{clip_id}.wav", local_dir=args.data_dir)
        )
        if clip_rate != rate:
            raise ValueError(f"{clip_id}: {clip_rate} Hz against the talk's {rate} Hz")
        span = locate_clip(clip, talk, rate)
        if span is None:
            failed.append(iid)
            continue
        spans[f"mcif_{iid}"] = [[round(span.start_s, 2), round(span.end_s, 2)]]
        widths.append(span.end_s - span.start_s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spans, indent=1), encoding="utf-8")
    print(f"{len(spans)} of {len(answerable)} answerable questions located -> {args.out}")
    if widths:
        print(
            f"  span seconds: min {min(widths):.1f}, median {statistics.median(widths):.1f}, "
            f"max {max(widths):.1f}"
        )
    for iid in failed:
        print(f"  not located: {iid}")


if __name__ == "__main__":
    main()
