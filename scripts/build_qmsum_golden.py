"""Adapter: QMSum on AMI meeting audio -> a time-span golden set.

QMSum (arXiv 2104.05938) has human-written queries over AMI meeting transcripts,
each with its evidence marked as transcript turn indices. AMI times every word
of those transcripts. Matching each turn's words, in order, against its
speaker's timed words turns the human turn labels into seconds, the unit
recordings are scored in (src/eval/spans.py). The labels stay human; only their
unit changes, through AMI's forced-alignment word timings.

Only specific queries are kept: a general query's evidence is the whole
meeting, which retrieval within that meeting cannot miss. Meetings whose turns
do not match cleanly are dropped, and so are meetings AMI lists with bad
timings or an audio offset.

Usage:
    uv run python -m scripts.fetch_ami --split test
    uv run python -m scripts.build_qmsum_golden --split test
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.types.eval import GoldenQuery, GoldenSet, TimeSpan

# AMI role codes in corpusResources/meetings.xml -> QMSum speaker names.
_ROLES = {
    "PM": "Project Manager",
    "ME": "Marketing",
    "UI": "User Interface",
    "ID": "Industrial Designer",
}
# groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml: word timings bad or
# incomplete (TS3009c), audio starting minutes after the transcript (TS3011d).
_EXCLUDED = {"TS3009c", "TS3011d"}
# A turn is located when its first words appear this close after the previous
# match in the speaker's stream.
_SEARCH_WINDOW = 200
_ANCHOR_WORDS = 3
_NITE = "{http://nite.sourceforge.net/}"


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class MatchStats:
    turns: int
    with_words: int
    exact: int

    @property
    def rate(self) -> float:
        return self.exact / self.with_words if self.with_words else 0.0


def speaker_letters(meetings_xml: str, meeting: str) -> dict[str, str]:
    """QMSum speaker name -> AMI speaker letter for one meeting."""
    # AMI's own annotation release. ElementTree resolves no external entities,
    # and the bundled Expat caps entity expansion.
    root = ET.fromstring(meetings_xml.encode("iso-8859-1"))
    for node in root.iter("meeting"):
        if node.get("observation") == meeting:
            return {_ROLES[s.get("role", "")]: s.get("nxt_agent", "") for s in node.iter("speaker")}
    raise KeyError(f"{meeting} not in meetings.xml")


def read_words(path: Path) -> list[TimedWord]:
    """A speaker's timed words, punctuation included, markers left out."""
    root = ET.parse(path).getroot()
    return [
        TimedWord(
            text=el.text.lower(),
            start_s=float(el.get("starttime", 0)),
            end_s=float(el.get("endtime", 0)),
        )
        for el in root
        if el.tag == "w" and el.text and el.get("starttime") is not None
    ]


def _tokens(content: str) -> list[str]:
    return [t.lower() for t in content.split() if not re.fullmatch(r"\{.*\}", t)]


def turn_times(
    turns: Sequence[Mapping[str, str]],
    streams: Mapping[str, Sequence[TimedWord]],
    letters: Mapping[str, str],
) -> tuple[list[tuple[float, float] | None], MatchStats]:
    """(start, end) seconds per turn, None for a turn with no words or no match.

    Each speaker's stream is consumed in order, so a turn is matched only after
    that speaker's previous one; an unmatched turn leaves the position as it was.
    """
    position = dict.fromkeys(streams, 0)
    times: list[tuple[float, float] | None] = []
    with_words = exact = 0
    for turn in turns:
        letter = letters[turn["speaker"]]
        tokens = _tokens(turn["content"])
        if not tokens:
            times.append(None)
            continue
        with_words += 1
        stream, start = streams[letter], position[letter]
        k = min(_ANCHOR_WORDS, len(tokens))
        hit = next(
            (
                j
                for j in range(start, min(start + _SEARCH_WINDOW, len(stream) - k + 1))
                if [w.text for w in stream[j : j + k]] == tokens[:k]
            ),
            None,
        )
        window = stream[hit : hit + len(tokens)] if hit is not None else []
        if hit is None or [w.text for w in window] != tokens:
            times.append(None)
            continue
        exact += 1
        position[letter] = hit + len(tokens)
        times.append((window[0].start_s, max(w.end_s for w in window)))
    return times, MatchStats(turns=len(turns), with_words=with_words, exact=exact)


def query_spans(
    spans: Sequence[Sequence[str]], times: Sequence[tuple[float, float] | None]
) -> list[TimeSpan]:
    """Each labelled turn range as the time from its first to its last located turn."""
    out: list[TimeSpan] = []
    for first, last in spans:
        located = [t for t in times[int(first) : int(last) + 1] if t is not None]
        if located:
            out.append(
                TimeSpan(start_s=min(a for a, _ in located), end_s=max(b for _, b in located))
            )
    return out


def to_queries(
    meeting: str, split: str, qmsum: Mapping[str, Any], times: Sequence[tuple[float, float] | None]
) -> list[GoldenQuery]:
    queries: list[GoldenQuery] = []
    for i, q in enumerate(qmsum["specific_query_list"]):
        spans = query_spans(q["relevant_text_span"], times)
        if not spans:
            continue
        n_turns = sum(int(b) - int(a) + 1 for a, b in q["relevant_text_span"])
        seconds = sum(s.end_s - s.start_s for s in spans)
        queries.append(
            GoldenQuery(
                query_id=f"qmsum_{meeting}_s{i}",
                text=q["query"].strip(),
                paper_id=meeting,
                category="factual",
                relevant_spans=spans,
                expected_facts=[q["answer"].strip()] if q.get("answer") else [],
                note=f"QMSum | split={split} | turns={n_turns} | seconds={seconds:.0f}",
            )
        )
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ami"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--min-match", type=float, default=0.95)
    parser.add_argument("--output", type=Path, help="default data/golden/qmsum-ami-<split>-v1.yaml")
    args = parser.parse_args()

    ann = args.data_dir / "annotations"
    meetings_xml = (ann / "corpusResources" / "meetings.xml").read_text(encoding="iso-8859-1")
    files = sorted((args.data_dir / "qmsum" / args.split).glob("*.json"))
    if not files:
        raise SystemExit(
            f"No QMSum files under {args.data_dir / 'qmsum' / args.split}; run fetch_ami"
        )

    queries: list[GoldenQuery] = []
    dropped: list[str] = []
    rates: list[float] = []
    for path in files:
        meeting = path.stem
        if meeting in _EXCLUDED:
            dropped.append(f"{meeting} (AMI data problem)")
            continue
        qmsum = json.loads(path.read_text(encoding="utf-8"))
        letters = speaker_letters(meetings_xml, meeting)
        streams = {
            letter: read_words(ann / "words" / f"{meeting}.{letter}.words.xml")
            for letter in letters.values()
        }
        times, stats = turn_times(qmsum["meeting_transcripts"], streams, letters)
        rates.append(stats.rate)
        if stats.rate < args.min_match:
            dropped.append(f"{meeting} (match {stats.rate:.2f})")
            continue
        queries.extend(to_queries(meeting, args.split, qmsum, times))

    output = args.output or Path(f"data/golden/qmsum-ami-{args.split}-v1.yaml")
    golden = GoldenSet(name="qmsum-ami", version=f"{args.split}-v1", queries=queries)
    output.write_text(
        yaml.safe_dump(
            golden.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
        ),
        encoding="utf-8",
    )
    seconds = sorted(sum(s.end_s - s.start_s for s in q.relevant_spans) for q in queries)
    print(
        f"{len(files)} meetings, {len(files) - len(dropped)} kept, {len(queries)} queries -> {output}"
    )
    print(f"  turn match rate: min {min(rates):.3f}, mean {sum(rates) / len(rates):.3f}")
    if seconds:
        print(
            "  evidence seconds per query: "
            f"p25 {seconds[len(seconds) // 4]:.0f}, median {seconds[len(seconds) // 2]:.0f}, "
            f"p75 {seconds[3 * len(seconds) // 4]:.0f}"
        )
    for d in dropped:
        print(f"  dropped {d}")


if __name__ == "__main__":
    main()
