"""Build the local span-labelling page for MCIF golden candidates.

For each talk with answerable candidates the page shows the video, its slide
keyframes, the timestamped transcript, and each question with MCIF's reference
answer. It suggests no spans: a human marks every one. The page keeps progress
in the browser and exports `mcif-spans.json`, which
`build_mcif_golden --spans` merges into the candidates.

Usage:
    uv run python -m scripts.build_label_sheet \\
        --candidates data/golden/_candidates/mcif-v1.yaml \\
        --pages-dir data/mcif/pages --media-dir data/mcif/MCIF_DATA/LONG_VIDEOS \\
        --out data/mcif/label/index.html
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.ingestion.media import Word, keyframe_path, load_manifest, manifest_path, words_path
from src.types.eval import GoldenQuery

_TEMPLATE = Path(__file__).with_name("label_sheet.html")
# A transcript line ends at a sentence end once it has this many words, or
# at the cap, so a line is short enough to click to the right moment.
_MIN_LINE_WORDS = 6
_MAX_LINE_WORDS = 24


def transcript_lines(words: Sequence[Word]) -> list[dict[str, Any]]:
    """Words grouped into short timed lines for the transcript panel."""
    lines: list[dict[str, Any]] = []
    group: list[Word] = []
    for word in words:
        group.append(word)
        sentence_end = word.text.rstrip().endswith((".", "?", "!"))
        if (sentence_end and len(group) >= _MIN_LINE_WORDS) or len(group) >= _MAX_LINE_WORDS:
            lines.append(_line(group))
            group = []
    if group:
        lines.append(_line(group))
    return lines


def _line(group: Sequence[Word]) -> dict[str, Any]:
    return {
        "start": round(group[0].start_s, 2),
        "end": round(group[-1].end_s, 2),
        "text": "".join(w.text for w in group).strip(),
    }


def _rel(target: Path, base: Path) -> str:
    return os.path.relpath(target, base).replace(os.sep, "/")


def build_data(
    candidates: Sequence[GoldenQuery], pages_dir: Path, media_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Everything the page renders, for talks that have been ingested."""
    by_talk: dict[str, list[GoldenQuery]] = {}
    for q in candidates:
        if q.category != "out_of_corpus":
            by_talk.setdefault(q.paper_id, []).append(q)

    talks: list[dict[str, Any]] = []
    missing: list[str] = []
    for talk, questions in sorted(by_talk.items()):
        if not manifest_path(pages_dir, talk).exists():
            missing.append(talk)
            continue
        manifest = load_manifest(pages_dir, talk)
        raw = json.loads(words_path(pages_dir, talk).read_text(encoding="utf-8"))["words"]
        words = [Word(start_s=float(a), end_s=float(b), text=str(t)) for a, b, t in raw]
        talks.append(
            {
                "id": talk,
                "video": _rel(media_dir / manifest.source, out_dir),
                "duration": manifest.duration_s,
                "segments": [
                    {
                        "page": s.page,
                        "start": s.start_s,
                        "end": s.end_s,
                        "img": _rel(keyframe_path(pages_dir, talk, s.page), out_dir),
                    }
                    for s in manifest.segments
                ],
                "lines": transcript_lines(words),
                "questions": [
                    {
                        "id": q.query_id,
                        "text": q.text,
                        "reference": q.expected_facts[0] if q.expected_facts else "",
                        "tag": (q.note or "").removeprefix("MCIF | "),
                    }
                    for q in questions
                ],
            }
        )
    return {"talks": talks, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/golden/_candidates/mcif-v1.yaml")
    )
    parser.add_argument("--pages-dir", type=Path, default=Path("data/mcif/pages"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/mcif/MCIF_DATA/LONG_VIDEOS"))
    parser.add_argument("--out", type=Path, default=Path("data/mcif/label/index.html"))
    args = parser.parse_args()

    raw = yaml.safe_load(args.candidates.read_text(encoding="utf-8")) or []
    candidates = [GoldenQuery.model_validate(d) for d in raw]
    data = build_data(candidates, args.pages_dir, args.media_dir, args.out.parent)
    # "</" inside the inline JSON would close the <script> element early.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = _TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    n_questions = sum(len(t["questions"]) for t in data["talks"])
    print(f"{len(data['talks'])} talks, {n_questions} questions -> {args.out}")
    if data["missing"]:
        print(f"  not ingested yet, left out: {', '.join(data['missing'])}")


if __name__ == "__main__":
    main()
