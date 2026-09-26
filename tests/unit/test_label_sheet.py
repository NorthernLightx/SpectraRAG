"""The span-labelling page's data: talks, keyframes, transcript lines, and no spans."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_label_sheet import build_data, transcript_lines
from src.ingestion.media import MediaManifest, Word, save_words, write_manifest
from src.types import MediaSegment
from src.types.eval import GoldenQuery


def test_transcript_lines_break_at_sentence_ends() -> None:
    words = [Word(float(i), float(i) + 0.5, f" w{i}") for i in range(5)]
    words += [Word(5.0, 5.5, " end."), Word(6.0, 6.5, " next")]
    lines = transcript_lines(words)
    assert [line["text"] for line in lines] == ["w0 w1 w2 w3 w4 end.", "next"]
    assert (lines[0]["start"], lines[0]["end"]) == (0.0, 5.5)


def test_build_data_lists_ingested_talks_and_skips_unanswerable(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    write_manifest(
        pages,
        MediaManifest(
            doc_id="talk",
            source="talk.mp4",
            duration_s=20.0,
            transcriber="fake",
            segmentation={},
            segments=[MediaSegment(page=1, start_s=0.0, end_s=20.0)],
        ),
    )
    save_words(pages, "talk", "fake", [Word(1.0, 1.5, " Hello.")])
    candidates = [
        GoldenQuery(
            query_id="a",
            text="Q?",
            paper_id="talk",
            category="factual",
            expected_facts=["A."],
            note="MCIF | qa_type=AV | origin=General",
        ),
        GoldenQuery(query_id="b", text="GPU?", paper_id="talk", category="out_of_corpus"),
        GoldenQuery(
            query_id="c", text="Q2?", paper_id="other", category="factual", expected_facts=["B."]
        ),
    ]
    data = build_data(candidates, pages, tmp_path / "videos", tmp_path / "label")
    assert data["missing"] == ["other"]
    [talk] = data["talks"]
    assert talk["video"] == "../videos/talk.mp4"
    assert talk["segments"][0]["img"] == "../pages/talk/talk_p1.png"
    assert [q["id"] for q in talk["questions"]] == ["a"]
    assert talk["questions"][0]["tag"] == "qa_type=AV | origin=General"
    assert "span" not in json.dumps(talk["questions"])
