"""MCIF -> golden candidates: the join, prompt and category rules.

The join is the load-bearing one. Questions live in the parquet and answers in
the reference XML, matched by sample id; a wrong join pairs every question with
another question's answer while every count still looks right.
"""

from __future__ import annotations

import pytest

from scripts.build_mcif_golden import _category, _question, to_candidates
from scripts.fetch_mcif import qa_samples

_PREFIX = "Answer the following question concisely given the English content: "

_XML = """<?xml version='1.0' encoding='utf-8'?>
<testset name="MCIF1.2">
  <task track="long" text_lang="en">
    <sample id="0" iid="QA_13_135" task="QA" qa_type="AV" qa_origin="Transcript">
      <audio_path>ICWfTnUMio.wav</audio_path>
      <video_path>ICWfTnUMio.mp4</video_path>
      <reference>They are trained on large-scale web-crawled data.</reference>
    </sample>
    <sample id="2" iid="SUM_52" task="SUM">
      <video_path>crgYiwKDfX.mp4</video_path>
      <reference>A summary.</reference>
    </sample>
    <sample id="7" iid="QA_13_140" task="QA" qa_type="NA" qa_origin="General">
      <video_path>ICWfTnUMio.mp4</video_path>
      <reference>Not answerable.</reference>
    </sample>
    <sample id="9" iid="QA_20_203" task="QA" qa_type="V" qa_origin="General">
      <video_path>krJSAnVcGR.mp4</video_path>
      <reference>McGill University/Mila and Microsoft Research.</reference>
    </sample>
  </task>
</testset>
"""

_PROMPTS = {
    "0": _PREFIX + "What are the main data sources for language models?",
    "7": _PREFIX + "Which GPU did the authors use?",
    "9": _PREFIX + "What are the affiliations of the authors of the paper?",
}


def test_qa_samples_keeps_only_qa_and_reads_the_talk_id() -> None:
    samples = qa_samples(_XML)
    assert [s.iid for s in samples] == ["QA_13_135", "QA_13_140", "QA_20_203"]
    assert samples[0].talk == "ICWfTnUMio"
    assert samples[2].reference == "McGill University/Mila and Microsoft Research."


def test_question_strips_the_fixed_prompt() -> None:
    assert _question(_PROMPTS["0"]) == "What are the main data sources for language models?"


def test_question_fails_loudly_on_an_unexpected_prompt() -> None:
    with pytest.raises(ValueError):
        _question("Summarise the talk.")


def test_category_mapping() -> None:
    assert _category("V") == "figure"
    assert _category("NA") == "out_of_corpus"
    assert _category("AV") == "factual"
    assert _category("A") == "factual"


def test_candidates_pair_each_question_with_its_own_answer() -> None:
    by_id = {q.query_id: q for q in to_candidates(qa_samples(_XML), _PROMPTS)}
    q = by_id["mcif_QA_20_203"]
    assert q.text == "What are the affiliations of the authors of the paper?"
    assert q.paper_id == "krJSAnVcGR"
    assert q.expected_facts == ["McGill University/Mila and Microsoft Research."]
    assert q.relevant_spans == []


def test_unanswerable_candidate_has_no_facts() -> None:
    by_id = {q.query_id: q for q in to_candidates(qa_samples(_XML), _PROMPTS)}
    q = by_id["mcif_QA_13_140"]
    assert q.category == "out_of_corpus"
    assert q.expected_facts == []


def test_candidates_fail_loudly_on_a_missing_question() -> None:
    with pytest.raises(KeyError):
        to_candidates(qa_samples(_XML), {"0": _PROMPTS["0"]})
