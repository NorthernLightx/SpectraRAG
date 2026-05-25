"""Unit tests for the official MMLongBench-Doc QA scorer (scripts/experiments/
score_mmlb_qa.py). Covers the rule-based matcher per answer format, the
ACC/F1 aggregation (incl. the unanswerable negative-class split), and the
extractor-reply parser. No Ollama is touched — only the pure scoring code.

Expected scores are hand-derived against the official eval/eval_score.py logic
(github.com/mayubo2333/MMLongBench-Doc) and asserted exactly. ANLS is
continuous, so Str/List cases assert the fractional value, not just truthiness.
"""

from __future__ import annotations

import math

import pytest

from scripts.experiments.score_mmlb_qa import (
    _acc_and_f1,
    _anls_compute,
    _eval_score,
    _gold_format,
    parse_extracted,
)


# --------------------------------------------------------------------------
# Int
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gt,pred,expected",
    [
        ("21", "21", 1.0),
        ("21", "21.0", 1.0),  # int(float(pred)) path
        ("9", "9", 1.0),
        ("21", "22", 0.0),
        ("21", "twenty-one", 0.0),  # unparseable -> pred="" -> 0
        ("8980", "8980", 1.0),
    ],
)
def test_int_matching(gt: str, pred: str, expected: float) -> None:
    assert _eval_score(gt, pred, "Int") == expected


# --------------------------------------------------------------------------
# Float — percentage scaling + closeness tolerance
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gt,pred,expected",
    [
        ("21%", "21", 1.0),  # trailing % stripped both sides
        ("0.45", "0.45", 1.0),
        ("92%", "0.92", 1.0),  # include_percentage: 92/100 matches 0.92
        ("92", "92%", 1.0),
        ("7.0", "7.05", 1.0),  # isclose rel_tol=0.01: 0.05/7 = 0.7% < 1%
        ("7.0", "8.0", 0.0),
        ("453.25", "453.25", 1.0),
        ("2.5", "abc", 0.0),  # unparseable pred
    ],
)
def test_float_matching(gt: str, pred: str, expected: float) -> None:
    assert _eval_score(gt, pred, "Float") == expected


# --------------------------------------------------------------------------
# Str — ANLS (continuous) and the exact-match override
# --------------------------------------------------------------------------
def test_str_exact_and_case() -> None:
    assert _eval_score("Europe", "Europe", "Str") == 1.0
    assert _eval_score("Europe", "europe", "Str") == 1.0  # lowercased in clean
    assert _eval_score("Yellow", "Yellow", "Str") == 1.0


def test_str_anls_is_continuous() -> None:
    # "men" vs "women": Levenshtein 2, max len 5 -> ANLS = 1 - 2/5 = 0.6 (> 0.5).
    score = _eval_score("men", "women", "Str")
    assert math.isclose(score, 0.6, abs_tol=1e-9)


def test_str_anls_below_threshold_is_zero() -> None:
    # Very different strings: ANLS <= 0.5 collapses to 0.0.
    assert _eval_score("Clinton", "Trump", "Str") == 0.0


def test_str_exact_match_path_blocks_partial_credit() -> None:
    # "page N" hits is_exact_match -> requires ==, so a near-miss scores 0
    # even though raw ANLS("page 9","page 10") would be > 0.5.
    assert _eval_score("page 9", "page 9", "Str") == 1.0
    assert _eval_score("page 9", "page 10", "Str") == 0.0
    raw = _anls_compute("page 9", "page 10")
    assert raw > 0.5  # confirms the override actually changed the outcome


# --------------------------------------------------------------------------
# None == unanswerable gold, scored with the Str path
# --------------------------------------------------------------------------
def test_none_format_uses_str_path() -> None:
    assert _eval_score("Not answerable", "Not answerable", "None") == 1.0
    assert _eval_score("Not answerable", "Europe", "None") == 0.0


# --------------------------------------------------------------------------
# List — set-style matching, length gate, float-list joined equality
# --------------------------------------------------------------------------
def test_list_order_insensitive_match() -> None:
    # Sorted before comparison, so element order is ignored.
    assert _eval_score("['Circle', 'Rectangle']", "['Rectangle', 'Circle']", "List") == 1.0


def test_list_length_mismatch_is_zero() -> None:
    assert _eval_score("['a', 'b']", "['a']", "List") == 0.0
    assert _eval_score("['a', 'b', 'c']", "['a', 'b']", "List") == 0.0


def test_list_numeric_joined_equality() -> None:
    # First sorted element is float-like -> joined "-".join exact equality.
    assert _eval_score("['3', '1']", "['1', '3']", "List") == 1.0
    assert _eval_score("['3', '1']", "['3', '2']", "List") == 0.0


def test_list_string_elements_use_min_anls() -> None:
    # Non-float elements: score is min ANLS across aligned (sorted) pairs.
    # ['apple','banana'] vs ['apple','bananas']: apple==apple (1.0),
    # banana vs bananas -> Lev 1, maxlen 7 -> ANLS 1-1/7 ~= 0.857. min -> 0.857.
    score = _eval_score("['apple', 'banana']", "['apple', 'bananas']", "List")
    assert math.isclose(score, 1 - 1 / 7, abs_tol=1e-9)


def test_list_scalar_pred_wrapped() -> None:
    # A non-bracketed pred is wrapped to a 1-element list; vs a 1-element gold
    # list it can still match.
    assert _eval_score("['Pyke']", "Pyke", "List") == 1.0


# --------------------------------------------------------------------------
# Aggregation: ACC and the unanswerable-aware F1
# --------------------------------------------------------------------------
def _s(answer: str, pred: str, score: float) -> dict[str, object]:
    return {"answer": answer, "pred": pred, "score": score}


def test_acc_and_f1_mixed_set() -> None:
    # 3 answerable (2 right, 1 wrong) + 2 unanswerable (1 refused, 1 answered).
    samples = [
        _s("Europe", "Europe", 1.0),
        _s("Asia", "Asia", 1.0),
        _s("21", "22", 0.0),
        _s("Not answerable", "Not answerable", 1.0),
        _s("Not answerable", "Berlin", 0.0),
    ]
    acc, f1 = _acc_and_f1(samples)
    # ACC = 3/5
    assert math.isclose(acc, 0.6, abs_tol=1e-9)
    # recall = 2/3 (correct over answerable golds)
    # precision = 2/4 (numerator over pred!="Not answerable": s1,s2,s3,s5)
    # f1 = 2 * (2/3) * 0.5 / (2/3 + 0.5) = 4/7
    assert math.isclose(f1, 4 / 7, abs_tol=1e-9)


def test_f1_hallucination_penalty() -> None:
    # Two identical-correctness sets differ only in whether the model answers
    # an unanswerable question. Answering it (hallucinating) must lower F1.
    refused = [_s("X", "X", 1.0), _s("Not answerable", "Not answerable", 1.0)]
    hallucinated = [_s("X", "X", 1.0), _s("Not answerable", "wrong", 0.0)]
    _, f1_refused = _acc_and_f1(refused)
    _, f1_hall = _acc_and_f1(hallucinated)
    assert f1_refused == 1.0  # recall=precision=1
    assert f1_hall < f1_refused  # precision drops: denom grows, numerator same


def test_f1_all_unanswerable_is_zero() -> None:
    # No answerable golds -> recall denominator 0 -> F1 0.0 (ACC still defined).
    acc, f1 = _acc_and_f1([_s("Not answerable", "Not answerable", 1.0)])
    assert acc == 1.0
    assert f1 == 0.0


def test_acc_and_f1_empty() -> None:
    assert _acc_and_f1([]) == (0.0, 0.0)


# --------------------------------------------------------------------------
# Extractor-reply parsing + golden format extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Extracted answer: 21%\nAnswer format: Float", "21%"),
        ("Extracted answer: Not answerable\nAnswer format: String", "Not answerable"),
        ("Extracted answer: ['a', 'b']\nAnswer format: List", "['a', 'b']"),
        ("  Extracted answer:   Europe  \nAnswer format: String", "Europe"),
        ("21", "21"),  # no label -> whole reply
    ],
)
def test_parse_extracted(reply: str, expected: str) -> None:
    assert parse_extracted(reply) == expected


@pytest.mark.parametrize(
    "note,expected",
    [
        ("MMLongBench-Doc | doc_type=X | answer_format=Int", "Int"),
        ("MMLongBench-Doc | answer_format=None", "None"),
        ("MMLongBench-Doc | answer_format=List", "List"),
        (None, "Str"),  # missing note defaults to Str
        ("no format here", "Str"),
    ],
)
def test_gold_format(note: str | None, expected: str) -> None:
    assert _gold_format(note) == expected
