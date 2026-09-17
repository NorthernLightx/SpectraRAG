# ADR 0032: Routing is a cost lever, not an accuracy lever

Status: accepted (supersedes [0013](./0013-routing-is-the-accuracy-lever.md))

## Context

ADR 0013 concluded that per-query routing is where retrieval accuracy comes
from, and the README's headline (+35 % recall@10) rests on it. That conclusion
was drawn on MMLongBench-Doc: 20 documents, 107 in-corpus queries, and a
confidence interval of roughly ±0.06. At that width a +0.05 effect is
undetectable, so "routing is the lever" was consistent with the data without
being tested against the obvious alternative: always running both legs.

MMDocIR ([2501.08828](https://arxiv.org/abs/2501.08828)) has the power the old
set lacked: 1,127 queries over 218 documents and 4,837 pages at
`--page-cap 60`, with page **and** bounding-box evidence labels. Its documents
overlap MMLongBench-Doc's, so every number below is reported on the 1,029
queries whose documents are absent from the committed MMLongBench baseline
corpus. Contamination turned out not to matter: the 98 overlapping queries score 0.759
against 0.800 clean, slightly *worse*, so there is no tuning advantage to
subtract.

## Decision

Fuse text and visual on every query. Keep the classifier, but treat it as a
switch for saving work rather than for gaining accuracy. On this hardware it
does not currently save any.

## What was measured

Retrieval, recall@10, n=1,029, identical corpus and reranker across arms:

| arm | recall@10 | median latency |
|---|---|---|
| text-only | 0.460 ±0.030 | 4,265 ms |
| classifier router | 0.621 ±0.030 | 5,860 ms |
| **always-hybrid** | **0.784 ±0.025** | 5,843 ms |
| visual-only | 0.800 ±0.024 | 983 ms |

Always-hybrid beats the classifier by **+0.163 (+26 %)**, better on 191 queries
and worse on 3. The loss is entirely in the routing decision: on the 555
queries the classifier sends to the text leg it scores 0.436 where the visual
leg alone reaches 0.784, while on the 572 it routes to hybrid it matches
visual-only to within 0.010.

**The cost defence fails too.** Router and always-hybrid have the same median
latency, 5,860 ms against 5,843 ms. The reranked text leg runs on every query
either way and accounts for ~4.3 s of that; once pages are pre-encoded (ADR
0028) the visual leg's marginal cost is close to zero. Routing therefore buys
half the recall for the same wall clock.

Answer correctness, n=150 stratified, `gemma-4-26b` reading, graded with
`judge_answer_correctness`:

| subset | n | always-hybrid | router | p |
|---|---|---|---|---|
| retrieval differs | 18 | **0.333** | 0.000 | 0.016 |
| retrieval identical | 132 | 0.294 | 0.317 | 0.18 |

The retrieval gain does convert, but only on the ~12 % of queries where routing
changes what is retrieved; on the rest the arms receive identical context and
tie, which is the control. Full-set delta is +0.020 and not significant:
averaging over the 88 % that cannot differ buries the effect. Scaling the
discordant-slice gain to the corpus projects roughly **+0.06** absolute answer
correctness. That is a projection from n=18, not a measurement.

## Three ways it could have been an artefact

Each of these could have explained the result away. None did:

- **Query mix.** Reweighting to 100 % factual questions still leaves
  visual-only ahead, 0.739 against text's 0.472. The corpus being
  figure-heavy is not the cause.
- **Corpus text density.** On the text-richest quartile of documents the gap is
  *widest* (0.833 against 0.508), so thin text layers are not the cause either.
- **Page budget.** The text leg's ten chunks collapse to 8.28 distinct pages
  while the visual leg returns ten. Handicapping the visual leg to eight pages
  costs it 0.019 of a 0.34 gap; at four pages it still scores 0.730.

## Consequences

- The shipped default should fuse both legs. `--force-route hybrid` exists on
  `eval_run` to measure it; the serving default is a separate change, landed in
  the amendment below.
- The README's "+35 %" remains true of the router against text-only, and is now
  the weaker of two available numbers.
- The reader, not retrieval, is the binding constraint: even handed the right
  page it answers correctly a third of the time. That agrees with
  [0025](./0025-structured-extraction-augments-reading.md).
- Untested and not claimed: prose corpora with no layout to exploit, corpora
  past ~5k pages, and per-query cost when every query needs a vision-capable
  reader.

## Amendment (2026-07-30): the reader's refusals are mostly correct

Chasing the reading bottleneck this ADR names, on the same 150-query subset with
`gemma-4-26b`:

- Of 120 queries whose correct evidence reached the reader, **46 % were refused**
  and only 11 % were answered wrongly. Refusal outnumbers misreading 4:1.
- A page image is worth no more than a text chunk once retrieved: 0.366 against
  0.388.
- Re-rendering pages from 72 DPI (what MMDocIR ships) to 150 DPI, 4.3x the
  pixels, moved correctness +0.023 and left the refusal rate at 49 %. Not the
  binding constraint.
- Telling the model that attached images count as context lifts the metric
  (+0.065, p=0.003) but **converts correct refusals into wrong answers**: 10 of
  20 newly-answered queries are wrong under the looser prompt, 23 of 29 under a
  variant that keeps the strict refusal wording. The failures are counting
  questions ("how many green bars appear in Figure 1"), which this reader cannot
  do from a page image.

No prompt variant shipped. `judge_answer_correctness` grades recall of expected
facts and ignores precision, so it scores a confident wrong answer above a
correct refusal. The metric moved while the product got worse. Receipts in
`data/eval/baseline-mmdocir-perception.json`.

Anything that trades refusals for attempts needs a precision-aware metric first.
The repo already has faithfulness judges; they were not part of this gate.

## Amendment (2026-09-17): the serving default now fuses

The measurement above did not reach the served retriever. Nothing read
`Settings.routing_mode`, and `RoutingMode` had no `hybrid` member, so no
configuration made the server fuse on every query. Only a per-call
`force_route="hybrid"` did, one request at a time. What shipped was the
classifier arm measured at 0.621 against always-hybrid's 0.784.

Three changes close that:

- `RoutingMode` gains `hybrid`. It runs both legs and skips the classifier.
  `force_route` still overrides it per call.
- `_wire_retriever_from_settings` passes `mode` and
  `cascade_confidence_threshold` to the `RoutingRetriever`. Asking for
  `cascade` without a threshold now raises at wiring time instead of being
  ignored.
- `Settings.routing_mode` defaults to `hybrid`. `category` and `cascade` stay
  reachable through `RAG_ROUTING_MODE`.

The `RoutingRetriever` constructor still defaults to `category`, which is what
`eval_run` builds without `--cascade`. Changing it would move the arm every
committed baseline was produced under. The hybrid arm stays addressable there
through `--force-route hybrid`.

Not measured: the latency table above comes from the local box against a
pre-encoded page index. Cloud Run serves the visual leg on CPU (ADR 0028), and
the classifier previously kept about half of traffic off it, so the served p50
under always-hybrid is unknown. Measure it against the live service before
reading the cost column as zero.

## Amendment (2026-09-17): the precision-aware metric exists, and it agrees

The 2026-07-30 amendment ended by saying anything that trades refusals for
attempts needs a precision-aware metric first. `answer_outcome` and
`outcome_rates` in `src/eval/metrics_generation.py` are that metric. Refusal is
read from the answer text, and only an attempt is graded, so a correct refusal
and a confident wrong answer stop collapsing to the same 0.0.

Replaying the four committed arms of
`data/eval/baseline-mmdocir-perception.json` through it, n=150 each:

| arm | mean coverage | refused | correct | wrong |
|---|---|---|---|---|
| hyb26b | 0.299 | 48.0 % | 22.7 % | 29.3 % |
| hyb26b_dpi150 | 0.322 | 48.0 % | 24.7 % | 27.3 % |
| hyb26b_vprompt | 0.387 | 31.3 % | 30.7 % | **38.0 %** |
| hyb26b_strict | 0.349 | 24.7 % | 25.3 % | **50.0 %** |

The prompt variant this ADR declined to ship lifts mean coverage by 0.088 and
raises wrong answers from 44 to 57 of 150. It converted 25 refusals, 13 of them
into wrong answers. The strict variant is worse: half its answers are wrong. The
no-ship call stands, now on a number rather than a reading of examples.

One caveat on the coverage column: every gen150 query carries exactly one
expected fact, so coverage admits only 0.0 and 1.0, and 21 to 28 grades per arm
sit off that grid. Dropping the off-grid grades moves each arm down 0.02 to
0.04 and leaves the ordering unchanged. `answer_correctness` now states the
legal grid in its prompt and flags returns that miss it.
