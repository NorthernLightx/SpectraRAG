"""Score an MMLongBench-Doc QA run with the OFFICIAL end-to-end protocol.

The repo's eval_run.py scores generation with a gold-answer substring match
(`gold_match`) plus an LLM judge (faithfulness / answer_relevance /
answer_correctness). Neither is the MMLongBench-Doc headline metric, so a number
from that path cannot be compared to the paper's leaderboard. This script
implements the paper's three-stage protocol so an "at/above SOTA" claim is
measured on the same ruler the authors used.

============================================================================
THE OFFICIAL PROTOCOL  (Ma et al., "MMLongBench-Doc: Benchmarking Long-context
Document Understanding with Visualizations", arXiv:2407.01523; reference code
github.com/mayubo2333/MMLongBench-Doc, files eval/extract_answer.py,
eval/prompt_for_answer_extraction.md, eval/eval_score.py — fetched 2026-05-24)
============================================================================

Three stages (README: "we adopt a three-stage evaluation protocol"):

  1. GENERATE   a free-form long answer (done upstream by eval_run.py; this
                script consumes that text, it does NOT call the RAG pipeline,
                so no ColQwen2 / GPU retrieval is triggered).

  2. EXTRACT    a short answer from the long answer with an LLM. The official
                extractor is GPT-4o (temperature 0); the repo routes LLM calls
                through Ollama, so the extractor here is a local text model
                (--extractor-model, default gemma3:4b). The prompt is the
                authors' verbatim eval/prompt_for_answer_extraction.md
                (vendored below as _EXTRACTION_PROMPT). The model must reply:
                    Extracted answer: [answer]
                    Answer format: [answer format]
                Two reserved strings: "Not answerable" (the analysis says the
                docs don't contain it) and "Fail to answer" (the analysis says
                it couldn't read the images). Both are parsed out and fed to
                stage 3 as the prediction string.

  3. SCORE      the extracted answer against the gold answer with rule-based
                matching keyed on the gold answer_format. This is a faithful
                port of eval/eval_score.py:eval_score — see _eval_score below,
                which carries the upstream logic line-for-line:

                - Int   : int(gt) == int(float(pred)); parse failure -> 0.
                - Float : is_float_equal(include_percentage=True, is_close=True)
                          -> pred matches any of {ref/100, ref, ref*100} within
                          isclose(rel_tol=0.01) OR rounded to
                          max(min(prec_pred, prec_gt), 2) decimals. "%" stripped.
                - Str / : get_clean_string both (lower, strip $ % quotes and
                  None    parentheses); if gt is_exact_match (URL, .py/.ipynb,
                          "page..."-prefix, phone, a.m./p.m., YYYY-MM[-DD],
                          email) require exact ==, else ANLS (1 - normalised
                          Levenshtein, thresholded: anls<=0.5 -> 0.0).
                - List  : parse bracketed literals to lists; LENGTH MISMATCH ->
                          0.0; sort cleaned elements; if first elem is
                          float-like or exact-match use joined "-".join==, else
                          min(ANLS) over the aligned pairs.

                Score is a FLOAT in [0,1] (ANLS is continuous), not a bool, so
                Str/List partial credit is preserved exactly as upstream.

AGGREGATE METRICS  (port of eval/eval_score.py:eval_acc_and_f1; the paper,
arXiv:2407.01523 sec. 4, "report both generalized accuracy and generalized F1
score to balance the answerable (positive) and unanswerable (negative)
questions"):

  ACC = mean(score) over all scored samples.   <- the "generalized accuracy"

  F1  = harmonic mean of recall and precision, where the gold/pred split is on
        the string "Not answerable" (the negative class):
          recall    = sum(score | gold != "Not answerable")
                      / count(gold != "Not answerable")
          precision = sum(score | gold != "Not answerable")
                      / count(pred != "Not answerable")
        Same numerator (correct credit on answerable golds). recall is hurt by
        wrongly refusing an answerable question; precision is hurt by answering
        an unanswerable one (the hallucination penalty: it grows the
        pred!="Not answerable" denominator without adding to the numerator).
        This is NOT token-level F1.

HEADLINE TO BEAT (paper Table 3): GPT-4o ACC 43.6 %, F1 42.7 %, on the full
1082-question benchmark. The user's "~44.9 %" is close to but not exactly the
published figure; the abstract/Table-3 value is F1 42.7 % / ACC 43.6 %. Note
this golden set is the 149-query in-corpus SUBSET, so a number here is NOT
directly the leaderboard number; it is the official metric computed on our
slice. State both the subset size and that caveat with any claim.

WHERE GOLD ANSWERS LIVE: data/golden/mmlongbench-v1.yaml. The gold short answer
is expected_facts[0]; the answer_format is parsed out of the `note` field
(`... | answer_format=Str|Int|Float|List|None`). answer_format==None is the
unanswerable class and aligns 1:1 with expected_facts==["Not answerable"] (36
of 149 queries). The MACHINE NEVER AUTHORS GROUND TRUTH: gold answers and
formats are the human MMLongBench labels read straight from the golden; this
script only runs the comparison.

CAVEAT — category vs answer_format: 40 golden queries carry category
`out_of_corpus` but only 36 carry answer_format==None. The 4 extra are queries
this repo's RETRIEVAL setup treats as out-of-corpus (their evidence pages are
outside the indexed slice) yet MMLongBench labels them answerable. The official
QA F1 keys on the MMLongBench answer ("Not answerable"), NOT on this repo's
`out_of_corpus` category, so those 4 count as ANSWERABLE here. --use-category
flips the breakdown label to the repo category if you want the repo-internal
view; the F1 split stays on the MMLongBench answer either way.

WHAT IS NOT REPRODUCED EXACTLY:
  - Extractor model: local gemma3:4b/llama3.2:3b, not GPT-4o. The matching
    rules are identical; only the free-text -> short-answer reduction differs.
    A weaker extractor can only lower the score vs GPT-4o, so a number here is
    a conservative lower bound on what the same generations would score under
    the official extractor. Swap --extractor-model to a stronger Ollama model
    to tighten it.
  - The extractor MUST run serially (Ollama serialises GPU work; the user's
    parallel ColQwen2 job shares the 8 GB card). This script issues one
    extraction at a time on purpose — do not add concurrency.

Usage:

    .venv/Scripts/python.exe -m scripts.experiments.score_mmlb_qa \
        --run data/eval/runs/exp_mmlb_gen_full.json \
        --golden data/golden/mmlongbench-v1.yaml \
        --answer-field text.answer \
        --extractor-model gemma3:4b \
        --cache data/eval/runs/mmlb_qa_extract_cache.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import sys
from collections import defaultdict
from math import isclose
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.experiments._openrouter_client import build_openrouter_client
from src.llm.ollama_chat import OllamaChatClient
from src.llm.protocol import LLMClient, Message

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# Reserved prediction strings from the official extractor (eval/prompt_for_
# answer_extraction.md). NA == "the docs don't contain it" (negative class);
# FAIL == "couldn't read the images" (scored as a wrong answer, never as a
# correct refusal).
_NOT_ANSWERABLE = "Not answerable"
_FAIL = "Fail to answer"

# Verbatim eval/prompt_for_answer_extraction.md (github.com/mayubo2333/
# MMLongBench-Doc, fetched 2026-05-24). The trailing example block ends with a
# bare "---"; the (question, analysis) pair is appended by the extractor as a
# separate `assistant` message (see _extract_one), matching eval/
# extract_answer.py exactly.
_EXTRACTION_PROMPT = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt.

---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\\n\\n1. Is the service safe?\\n2. Is the service effective?\\n3. Is the service caring?\\n4. Is the service responsive?\\n5. Is the service well-led?
Extracted answer: ['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\\n\\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\\n2. Regulation 12: Safe care and treatment\\n3. Regulation 18: Staffing\\n4. Regulation 11: Need for consent\\n5. Regulation 10: Dignity and respect\\n6. Regulation 9: Person-centred care\\n7. Regulation 17: Good governance\\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\\n\\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: Not answerable
Answer format: String

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: Fail to answer
Answer format: String

---"""

# answer_format=... lives inside the golden `note` string.
_FORMAT_RE = re.compile(r"answer_format=(\w+)")
# Parse the extractor's "Extracted answer: ..." line.
_EXTRACTED_RE = re.compile(r"Extracted answer:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


# --------------------------------------------------------------------------
# Stage 3: rule-based scorer. Faithful port of eval/eval_score.py.
# --------------------------------------------------------------------------
def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(distances[i1], distances[i1 + 1], distances_[-1]))
        distances = distances_
    return distances[-1]


def _anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    dist = _levenshtein(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def _is_float_equal(
    reference: Any,
    prediction: Any,
    include_percentage: bool = False,
    is_close: bool = False,
) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if "." in str(gt_ans):
            precision = len(str(gt_ans).split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    if include_percentage:  # noqa: SIM108 - kept as if/else to mirror upstream eval_score.py
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:  # noqa: SIM102 - nested if mirrors upstream eval_score.py
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def _get_clean_string(value: Any) -> str:
    s: str = str(value).lower().strip()
    # NOTE: upstream eval_score.py computes these rstrip()s but discards the
    # result (`s.rstrip(...)` with no reassignment). Reproduced as-is so scores
    # match the reference scorer rather than "fixing" a no-op.
    if s.endswith("mile"):
        s.rstrip("mile").strip()
    if s.endswith("miles"):
        s.rstrip("miles").strip()
    if s.endswith("million"):
        s.rstrip("million").strip()  # noqa: B005 - upstream no-op reproduced verbatim
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()  # remove parenthesis
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()  # remove surrounding quotes
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def _is_exact_match(s: str) -> bool:
    if "https://" in s:
        return True
    if s.endswith(".py") or s.endswith("ipynb"):
        return True
    if s.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s):  # telephone-number-like
        return True
    if "a.m." in s or "p.m." in s:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s):  # YYYY-MM-DD
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s):  # YYYY-MM
        return True
    # email — final clause kept explicit (not `return bool(...)`) to mirror the
    # upstream flag-accumulation structure of eval_score.py:is_exact_match.
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s):  # noqa: SIM103
        return True
    return False


def _isfloat(num: Any) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _safe_literal_list(s: str) -> Any:
    """Parse a bracketed string to a Python list. Upstream eval_score.py applies
    a bare Python expression-evaluator to these strings; we use
    ast.literal_eval so a malformed/hostile pred string can't execute code. On
    failure fall back to the raw string (so the len-mismatch branch yields 0.0
    instead of crashing), which is the conservative outcome."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _eval_score(gt: Any, pred: Any, answer_type: str) -> float:
    """Port of eval/eval_score.py:eval_score. Returns a float in [0, 1].

    `answer_type` is the MMLongBench answer_format: Int / Float / Str / None /
    List. Anything not in {Int, Float, Str, None} is treated as List, matching
    upstream's else-branch.
    """
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except Exception:
            pred = ""
        score: float = float(gt == pred)
    elif answer_type == "Float":
        try:
            gt = float(_get_clean_string(str(gt)))
            pred = float(_get_clean_string(str(pred)))
        except Exception:
            pred = ""
        score = float(_is_float_equal(gt, pred, include_percentage=True, is_close=True))
    elif answer_type in ("Str", "None"):
        gt = _get_clean_string(gt)
        pred = _get_clean_string(pred)
        if _is_exact_match(gt):  # noqa: SIM108 - kept as if/else to mirror upstream eval_score.py
            score = float(gt == pred)
        else:
            score = _anls_compute(gt, pred)
    else:  # List
        if isinstance(gt, str) and gt.startswith("["):
            gt = _safe_literal_list(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            pred = _safe_literal_list(pred)
        if not isinstance(pred, list):
            pred = [pred]
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted(_get_clean_string(a) for a in gt)
            pred = sorted(_get_clean_string(a) for a in pred)
            if _isfloat(gt[0]) or _is_exact_match(gt[0]):
                score = float("-".join(gt) == "-".join(pred))
            else:
                # lengths are equal here (the len() gate above returned 0.0
                # otherwise); strict=False mirrors upstream's plain zip().
                score = min(_anls_compute(g, p) for g, p in zip(gt, pred, strict=False))
    return float(score)


# --------------------------------------------------------------------------
# Aggregation: port of eval/eval_score.py:eval_acc_and_f1.
# --------------------------------------------------------------------------
def _acc_and_f1(samples: list[dict[str, Any]]) -> tuple[float, float]:
    """`samples` each have "score" (float), "answer" (gold short answer string),
    "pred" (extracted prediction string). Mirrors eval_acc_and_f1 exactly."""
    scored = [s for s in samples if "score" in s]
    if not scored:
        return 0.0, 0.0
    acc = sum(s["score"] for s in scored) / len(scored)
    try:
        pos = [s for s in scored if s["answer"] != _NOT_ANSWERABLE]
        attempted = [s for s in scored if s["pred"] != _NOT_ANSWERABLE]
        recall = sum(s["score"] for s in pos) / len(pos)
        precision = sum(s["score"] for s in pos) / len(attempted)
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0.0 else 0.0
    except ZeroDivisionError:
        f1 = 0.0
    return acc, f1


# --------------------------------------------------------------------------
# Golden / run plumbing.
# --------------------------------------------------------------------------
def _gold_format(note: str | None) -> str:
    if not note:
        return "Str"
    m = _FORMAT_RE.search(note)
    return m.group(1) if m else "Str"


def _dotted_get(record: dict[str, Any], field: str) -> Any:
    """Resolve a dotted answer field, e.g. 'text.answer' or 'answer_text'."""
    cur: Any = record
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def parse_extracted(reply: str) -> str:
    """Pull the short answer out of the extractor reply. The official format is
    `Extracted answer: <x>\\nAnswer format: <fmt>`. If the model omits the
    label we fall back to the whole stripped reply (the matcher then scores it
    as-is, which is the honest outcome for a malformed extraction)."""
    m = _EXTRACTED_RE.search(reply)
    if m:
        return m.group(1).strip()
    return reply.strip()


# Sentinel for an extraction that never succeeded after retries. The caller
# scores it as an empty prediction (0, never the "Not answerable" negative
# class) and does NOT cache it, so a later resume can retry — matching the
# spirit of upstream extract_answer.py's `except: response="Failed"`.
_EXTRACT_FAILED = "__extract_failed__"


async def _extract_one(
    client: LLMClient,
    model: str,
    question: str,
    analysis: str,
    *,
    max_attempts: int = 5,
) -> str:
    """One serial extraction. Message structure mirrors eval/extract_answer.py:
    prompt as the user turn, the (question, analysis) pair as the assistant
    turn, temperature 0.

    Works with either backend (both satisfy LLMClient):
    - Ollama can return a 500 when the shared GPU is momentarily out of VRAM (the
      user's parallel ColQwen2 job competes for the 8 GB card). OllamaChatClient
      only retries transport errors, not HTTP 5xx.
    - OpenRouter free-tier models 429 constantly ("...temporarily rate-limited
      upstream..."). OpenRouterClient already retries 429 internally (6 attempts
      to 60s) but on a heavily contended provider that budget can be exhausted
      and the 429 surfaces here.

    So we treat BOTH 429 and 5xx as transient: bounded backoff, then
    _EXTRACT_FAILED on exhaustion (caller leaves it uncached for a resume).
    Other 4xx (auth, model-not-found) are real config errors and fail fast."""
    messages = [
        Message(role="user", content=_EXTRACTION_PROMPT),
        Message(role="assistant", content=f"\n\nQuestion:{question}\nAnalysis:{analysis}\n"),
    ]
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.chat(messages, model=model, temperature=0.0, max_tokens=256)
            return parse_extracted(resp.text)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            transient = status == 429 or status >= 500
            if not transient:
                raise  # auth / bad request / model-not-found: not retryable.
            if attempt == max_attempts:
                return _EXTRACT_FAILED
            # Rate-limit window or VRAM pressure: back off and retry serially.
            await asyncio.sleep(min(2.0 * attempt, 15.0))
    return _EXTRACT_FAILED


def _print_breakdown(title: str, groups: dict[str, list[dict[str, Any]]]) -> None:
    print(f"  {title}")
    print(f"    {'group':<24}{'n':>5}{'ACC':>9}{'F1':>9}")
    for key in sorted(groups):
        bucket = groups[key]
        acc, f1 = _acc_and_f1(bucket)
        print(f"    {key:<24}{len(bucket):>5}{acc:>9.4f}{f1:>9.4f}")


async def run(args: argparse.Namespace) -> int:
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold_by_qid: dict[str, dict[str, Any]] = {}
    for q in golden["queries"]:
        facts = q.get("expected_facts") or [""]
        gold_by_qid[q["query_id"]] = {
            "answer": facts[0],
            "format": _gold_format(q.get("note")),
            "category": q.get("category", ""),
            "n_pages": len(q.get("relevant_pages") or []),
        }

    run_json = json.loads(args.run.read_text(encoding="utf-8"))
    per_query = run_json["per_query"] if isinstance(run_json, dict) else run_json

    cache: dict[str, str] = {}
    if args.cache and args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    # The extractor is text-only; either backend satisfies LLMClient and
    # _extract_one is provider-agnostic. The cache key carries the model name,
    # so a gemma3:4b cache and a deepseek-v4-flash:free cache never collide.
    client: LLMClient
    if args.extractor_provider == "openrouter":
        client = build_openrouter_client(timeout=args.timeout)
    else:
        client = OllamaChatClient(base_url=args.ollama_url)

    samples: list[dict[str, Any]] = []
    skipped_no_answer = 0
    skipped_no_gold = 0
    n = len(per_query)
    for i, record in enumerate(per_query):
        qid = record.get("query_id")
        gold = gold_by_qid.get(qid)
        if gold is None:
            skipped_no_gold += 1
            continue
        analysis = _dotted_get(record, args.answer_field)
        if not analysis or not str(analysis).strip():
            skipped_no_answer += 1
            continue

        question = record.get("query") or record.get("text") or ""
        cache_key = f"{args.extractor_model}::{args.answer_field}::{qid}"
        if cache_key in cache:
            pred = cache[cache_key]
        else:
            # Serial on purpose: Ollama serialises and the GPU is shared with a
            # parallel job. One extraction at a time. Do not parallelise.
            pred = await _extract_one(client, args.extractor_model, str(question), str(analysis))
            # Only cache a real extraction; a failed one stays uncached so a
            # later --cache resume retries it instead of freezing the failure.
            if pred != _EXTRACT_FAILED:
                cache[cache_key] = pred
                if args.cache:
                    args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"  [{i + 1}/{n}] {qid}  gold={gold['answer']!r}  pred={pred!r}", flush=True)

        # "Fail to answer" (couldn't read the doc) and a hard extraction failure
        # both map to an empty pred for matching: they score 0 and are NOT the
        # negative-class token, so they can't earn precision credit on an
        # unanswerable gold.
        pred_for_match = "" if pred in (_FAIL, _EXTRACT_FAILED) else pred
        score = _eval_score(gold["answer"], pred_for_match, gold["format"])
        gold_label = gold["category"] if args.use_category else gold["format"]
        samples.append(
            {
                "query_id": qid,
                "answer": gold["answer"],
                "pred": pred,
                "score": score,
                "format": gold["format"],
                "category": gold["category"],
                "n_pages": gold["n_pages"],
                "is_unanswerable": gold["answer"] == _NOT_ANSWERABLE,
                "gold_label": gold_label,
            }
        )

    if not samples:
        print(
            "No scorable (gold, answer) pairs. Check --answer-field and that the "
            "run shares query_ids with the golden."
        )
        return 2

    acc, f1 = _acc_and_f1(samples)
    print(f"\nMMLongBench-Doc OFFICIAL QA score  (run={args.run.name})")
    print(
        f"  extractor={args.extractor_model}  answer-field={args.answer_field!r}  "
        f"golden={args.golden.name}"
    )
    print(
        f"  scored={len(samples)}  skipped(no answer)={skipped_no_answer}  "
        f"skipped(no gold)={skipped_no_gold}"
    )
    print(f"  ACC = {acc:.4f}   ({acc * 100:.1f} %)")
    print(f"  F1  = {f1:.4f}   ({f1 * 100:.1f} %)")
    print("  (paper Table 3 GPT-4o on full 1082-q benchmark: ACC 43.6 % / F1 42.7 %;")
    print(
        f"   this is the OFFICIAL metric on the {len(samples)}-q in-corpus subset, "
        "not the leaderboard number.)\n"
    )

    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_format[s["format"]].append(s)
        by_category[s["category"]].append(s)
    _print_breakdown("By answer format:", by_format)
    print()
    _print_breakdown("By category:", by_category)

    # answerable / unanswerable split (the F1 axis), printed explicitly.
    answerable = [s for s in samples if not s["is_unanswerable"]]
    unanswerable = [s for s in samples if s["is_unanswerable"]]
    print()
    print("  Answerable vs unanswerable (the F1 axis):")
    print(
        f"    answerable    n={len(answerable):<4} mean score (recall) = "
        f"{_acc_and_f1(answerable)[0]:.4f}"
    )
    print(
        f"    unanswerable  n={len(unanswerable):<4} mean score          = "
        f"{_acc_and_f1(unanswerable)[0]:.4f}"
    )

    if args.scored_out:
        args.scored_out.write_text(json.dumps(samples, indent=2), encoding="utf-8")
        print(f"\n  Per-query scored samples -> {args.scored_out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="run JSON with generated answers")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    parser.add_argument(
        "--answer-field",
        default="answer_text",
        help="dotted field holding the generated answer text in each per_query record "
        "(e.g. 'answer_text' for EvalRun shape, 'text.answer' / 'vision.answer' for "
        "exp_mmlb_gen_full.json)",
    )
    parser.add_argument(
        "--extractor-provider",
        choices=("ollama", "openrouter"),
        default="ollama",
        help="backend for the extraction stage. 'ollama' (default) runs a local "
        "text model. 'openrouter' uses OpenRouterClient (key via "
        "RAG_OPENROUTER_API_KEY / .env) — use a stronger extractor like "
        "deepseek/deepseek-v4-flash:free, which avoids the gemma3:4b truncation "
        "confound (gemma3:4b cut off correct long answers, depressing scores).",
    )
    parser.add_argument(
        "--extractor-model",
        default="gemma3:4b",
        help="text model for the answer-extraction stage. For --extractor-provider "
        "ollama: an Ollama model (default gemma3:4b; try llama3.2:3b for faster/weaker). "
        "For openrouter: an OpenRouter text id, e.g. deepseek/deepseek-v4-flash:free. "
        "Runs SERIALLY either way.",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request HTTP timeout (s) for the openrouter extractor",
    )
    parser.add_argument(
        "--use-category",
        action="store_true",
        help="group/label by this repo's `category` instead of MMLongBench answer_format. "
        "Does NOT change the F1 negative-class split (always keyed on gold=='Not answerable').",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="JSON cache of extractions (resumable; keyed by model+field+qid)",
    )
    parser.add_argument(
        "--scored-out", type=Path, default=None, help="write per-query scored samples here"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print each (gold, pred) as it scores"
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
