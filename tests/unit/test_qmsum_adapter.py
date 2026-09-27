"""QMSum on AMI -> time-span golden queries: roles, turn timing, span conversion.

QMSum labels evidence as transcript turn indices; AMI times every word. The
conversion matches each turn's words, in order, against its speaker's timed
words. A wrong match moves every label while every count still looks right.
"""

from __future__ import annotations

import pytest

from scripts.build_qmsum_golden import (
    TimedWord,
    query_spans,
    speaker_letters,
    to_queries,
    turn_times,
)
from src.types.eval import TimeSpan

_MEETINGS_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
   <meeting nite:id="meet_49" observation="ES2004a">
      <speaker nite:id="ES2004a_2" channel="1" nxt_agent="B" role="PM"/>
      <speaker nite:id="ES2004a_3" channel="2" nxt_agent="C" role="ID"/>
      <speaker nite:id="ES2004a_1" channel="0" nxt_agent="A" role="UI"/>
      <speaker nite:id="ES2004a_4" channel="3" nxt_agent="D" role="ME"/>
   </meeting>
   <meeting nite:id="meet_50" observation="ES2004b">
      <speaker nite:id="ES2004b_2" channel="1" nxt_agent="A" role="PM"/>
   </meeting>
</nite:root>
"""


def test_speaker_letters_map_qmsum_roles_to_ami_letters() -> None:
    assert speaker_letters(_MEETINGS_XML, "ES2004a") == {
        "Project Manager": "B",
        "Industrial Designer": "C",
        "User Interface": "A",
        "Marketing": "D",
    }


def _stream(*words: tuple[str, float, float]) -> list[TimedWord]:
    return [TimedWord(text=w, start_s=s, end_s=e) for w, s, e in words]


STREAMS = {
    "A": _stream(("hmm", 0.4, 1.0), (".", 1.0, 1.0), ("yes", 20.0, 20.5), (".", 20.5, 20.5)),
    "B": _stream(("are", 5.0, 5.2), ("we", 5.2, 5.4), ("ready", 5.4, 6.1), ("?", 6.1, 6.1)),
}
LETTERS = {"User Interface": "A", "Project Manager": "B"}


def test_turns_are_timed_from_their_speakers_words_in_order() -> None:
    turns = [
        {"speaker": "User Interface", "content": "Hmm ."},
        {"speaker": "Project Manager", "content": "{vocalsound} Are we ready ?"},
        {"speaker": "User Interface", "content": "{disfmarker}"},
        {"speaker": "User Interface", "content": "Yes ."},
    ]
    times, stats = turn_times(turns, STREAMS, LETTERS)
    assert times == [(0.4, 1.0), (5.0, 6.1), None, (20.0, 20.5)]
    assert (stats.with_words, stats.exact) == (3, 3)


def test_an_unmatched_turn_is_untimed_and_does_not_derail_the_next() -> None:
    turns = [
        {"speaker": "User Interface", "content": "Something never said ."},
        {"speaker": "User Interface", "content": "Hmm ."},
    ]
    times, stats = turn_times(turns, STREAMS, LETTERS)
    assert times == [None, (0.4, 1.0)]
    assert (stats.with_words, stats.exact) == (2, 1)


def test_unknown_speaker_fails_loudly() -> None:
    with pytest.raises(KeyError):
        turn_times([{"speaker": "Visitor", "content": "Hi ."}], STREAMS, LETTERS)


def test_query_spans_cover_the_located_turns_of_each_span() -> None:
    times = [(0.0, 2.0), None, (5.0, 9.0), (10.0, 12.0), None]
    assert query_spans([["0", "2"], ["3", "4"]], times) == [
        TimeSpan(start_s=0.0, end_s=9.0),
        TimeSpan(start_s=10.0, end_s=12.0),
    ]
    assert query_spans([["1", "1"]], times) == []


def test_only_specific_queries_become_golden_queries() -> None:
    qmsum = {
        "general_query_list": [{"query": "Summarize the whole meeting.", "answer": "All of it."}],
        "specific_query_list": [
            {
                "query": "What did Marketing think of batteries?",
                "answer": "Too expensive.",
                "relevant_text_span": [["0", "0"]],
            },
            {
                "query": "What did the group say about colours?",
                "answer": "Yellow.",
                "relevant_text_span": [["1", "1"]],
            },
        ],
    }
    queries = to_queries("ES2004a", "test", qmsum, [(3.0, 8.0), None])
    assert [q.query_id for q in queries] == ["qmsum_ES2004a_s0"]
    q = queries[0]
    assert q.paper_id == "ES2004a"
    assert q.relevant_spans == [TimeSpan(start_s=3.0, end_s=8.0)]
    assert q.expected_facts == ["Too expensive."]
    assert q.note is not None and "split=test" in q.note and "turns=1" in q.note


def test_a_speaker_resyncs_after_a_long_unmatched_turn() -> None:
    # A long turn that fails to match leaves the speaker's position where it
    # was; the next turn sits hundreds of words ahead and must still be found.
    long_turn = [TimedWord(text=f"w{i}", start_s=float(i), end_s=i + 0.5) for i in range(600)]
    stream = {"A": [*long_turn, TimedWord("done", 700.0, 700.4), TimedWord(".", 700.4, 700.4)]}
    turns = [
        {
            "speaker": "User Interface",
            "content": " ".join(["w0", "w1", "w2", "x"] + [f"w{i}" for i in range(4, 600)]),
        },
        {"speaker": "User Interface", "content": "Done ."},
    ]
    times, _ = turn_times(turns, stream, {"User Interface": "A"})
    assert times[0] is None
    assert times[1] == (700.0, 700.4)
