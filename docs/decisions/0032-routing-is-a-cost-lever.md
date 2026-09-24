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
  and only 11 % were answered wrongly. Refusal outnumbers misreading 4:1
  (corrected on 2026-09-24, below).
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

## Amendment (2026-09-17): visual-only beats hybrid, by a tenth of the margin

The arms table above compares each arm to the classifier router, so the two
leading arms were never compared to each other. Their separate means, 0.800 and
0.784, sat inside overlapping intervals, which settles nothing: both arms answer
the same queries, so the comparison is paired and an unpaired interval is the
wrong instrument.

`scripts/experiments/paired_arm_compare.py` pairs the committed runs by
query_id, bootstraps the mean delta and runs an exact two-sided sign test over
the discordant pairs. It reads run JSONs only, so it needs no corpus and no
model. recall@10, n=1,127:

| pair | mean delta | 95 % CI | better / worse / tied | sign test |
|---|---|---|---|---|
| always-hybrid minus router | +0.1609 | [+0.1398, +0.1820] | 207 / 3 / 917 | p = 2e-57 |
| visual-only minus always-hybrid | +0.0158 | [+0.0043, +0.0278] | 44 / 20 / 1063 | p = 0.0037 |

Both are real. They are not the same size. Dropping the classifier is settled to
the point of not being worth re-testing, and it moves 207 queries. Dropping the
text leg would move 44 and cost 20, on 94 % ties.

The text leg stays, and the reason is not recall. Retrieval scoring counts gold
pages, and the text leg returns chunks, so it is what makes a citation point at
a passage rather than at a whole page. It is also the only leg that works on a
corpus with no page index built. Against that, ADR 0028's persisted index makes
the visual leg's marginal cost near zero while the reranked text leg accounts
for roughly 4.3 s of the 5.8 s median, so the text leg is the expensive half and
buys 0.0158 less recall. A deployment that wants latency over citations has a
measured case for visual-only, and `--force-route visual` already serves it.

What this does not license: reading the visual-only row of the regression gate
in `docs/results.md` as a recommendation. It passes the gate for the same reason
it wins here, and the gate cannot see what the text leg is kept for.

## Amendment (2026-09-23): at the chat's top-5, fusion costs more

The arms above were compared at recall@10. The chat asks for five results by
default, and at five the gap between always-hybrid and visual-only is wider.
Re-fusing the committed MMDocIR text and visual runs (all 1,127 queries,
`scripts/derive_arms.py --legs-from`):

| arm | recall@10 | recall@5 | nDCG@5 |
|---|---|---|---|
| visual-only | 0.797 | 0.748 | 0.708 |
| always-hybrid (w=1) | 0.784 | 0.692 | 0.552 |
| hybrid, w=5 | 0.797 | 0.719 | 0.609 |

Text pages pulled into the top five displace visual pages that were right.
Caveat: these legs come from two separate runs, and on 128 queries the
committed hybrid run's page set differs from what fusing them reproduces, so
leg-to-leg noise is mixed in. The clean measurement is one `eval_run --router
--force-route hybrid --fusion-depth 50` run, whose recorded legs give every arm
at every k from the same outputs. The served default is unchanged until that
run is in.

## Amendment (2026-09-24): with deeper legs, fusion loses at every weight

That run is in. One `eval_run --router --force-route hybrid --fusion-depth 50`
configuration ran over all 1,127 queries in four slices sharing one retrieval
fingerprint, merged into one run with both legs recorded 50 deep. The text leg
matches the committed hybrid run (bge-reranker-v2-m3 over 50 candidates);
ColQwen2 encodes queries in fp32 on CPU and the visual collection is searched
exactly. Every arm below comes from that run's legs, cut to the depth shown and
fused, so the arms differ only in depth and fusion. Receipt:
`data/eval/mmdocir-depth50-legs.json.gz`.

nDCG@5 here is the standard definition (`src/eval/metrics_retrieval.ndcg_at_k`,
ideal over min(gold pages, 5)). `derive_arms` scores a variant whose ideal counts
only the gold pages it retrieved; it reads about 0.02 higher and orders the arms
the same way. The previous amendment's nDCG@5 column is that variant.

Ten results, the size the tables above and the regression gate use:

| arm | recall@5 | nDCG@5 | recall@10 | MRR |
|---|---|---|---|---|
| visual-only | **0.749** | **0.686** | **0.796** | **0.689** |
| w=10, legs cut to 10 | 0.729 | 0.605 | 0.796 | 0.591 |
| w=2, legs cut to 10 | 0.715 | 0.582 | 0.796 | 0.567 |
| w=1, legs cut to 10 | 0.693 | 0.528 | 0.785 | 0.503 |
| w=10, legs at 50 | 0.719 | 0.598 | 0.794 | 0.588 |
| w=1, legs at 50 | 0.517 | 0.435 | 0.644 | 0.445 |
| text-only | 0.412 | 0.354 | 0.457 | 0.359 |

Five results with legs cut to five, the chat's default:

| arm | recall@5 | nDCG@5 | MRR |
|---|---|---|---|
| visual-only | **0.749** | **0.686** | **0.684** |
| w=2 or w=10 | 0.749 | 0.631 | 0.609 |
| w=1 | 0.730 | 0.554 | 0.512 |

Paired by query, visual-only leads every hybrid arm on nDCG@5 and MRR at both
sizes. At ten it leads w=1 by 0.055 recall@5 (95 % CI [0.039, 0.072]), 0.158
nDCG@5 and 0.186 MRR, and it leads the best hybrid arm, w=10 with legs cut to
ten, by 0.019 recall@5 (CI [0.009, 0.030]) and 0.081 nDCG@5 while tying recall@10
on every query. At five, any weight of 2 or more returns exactly visual-only's
page set, and w=1 trails it by 0.019 recall@5 (CI [0.006, 0.032]). What is left
is order: 0.055 nDCG@5 behind visual-only at w=2 and 0.132 at w=1. On the 1,029
queries outside the MMLongBench corpus every delta moves by less than 0.006.

Deeper legs make fusion worse, and reciprocal-rank fusion is the reason. With
k = 60, a page both legs rank 30th scores 2/90 = 0.022, more than a page only
the visual leg finds, at rank 1: 1/61 = 0.016. At depth 50, pages both legs
found fill 8 of w=1's top ten, against 3 at depth 10. On 227 queries w=1 at
depth 50 drops a gold page that visual-only keeps in its top ten, and 234 of
those 238 pages came from the visual leg alone, at a median visual rank of 1.
Raising the weight approaches visual-only from below and never passes it.

This run's legs cut to ten and fused at w=1 reproduce the committed hybrid run's
page sets on 1,066 queries, with recall@10 +0.004 (6 better, none worse). That
is consistent with the committed run's visual leg coming from the server's HNSW
graph where this run searched exactly, with query-encoding precision (the
committed visual run encoded in bf16 on GPU) behind other differences. The two
comparisons involve different pairs of runs, so their counts need not add up to
the previous amendment's 128. Server-backed visual stores now search exactly and
create new collections on disk without an HNSW graph (`src/rag/visual_store.py`).

Latency on the local box, p50: rerank 2.6 s, visual encode 1.25 s, visual
search 0.3 s, whole query 3.1 s. The legs run concurrently, and the reranker,
the largest stage, belongs to the text leg.

Consequences:

- `fusion_depth` above `top_k` makes every weight worse. It stays an
  instrument for recording legs, not a serving setting.
- On MMDocIR no weight or depth lets the text leg add page recall. At serving
  depth any weight above about 1.15 already returns the visual leg's page set.
- The served default stays at w=1 because the served corpus is the text-heavy
  arXiv demo, where ADR 0023 measured the opposite: the text leg ranks captions
  better, and every weight above 1 lost (6 of 15 figure and table queries worse,
  none better). The weight is per corpus. A visual-heavy corpus served with
  `RAG_VISUAL_FUSION_WEIGHT=2` gets visual-only's page set at the chat's five
  results and keeps the text leg's chunk on every page both legs found.

## Correction (2026-09-24): the refusal ratio depends on the grade and the depth

The 2026-07-30 amendment reports 46 % refused and 11 % wrong among 120
gold-present queries. The perception receipt does not record which pages
reached the reader, so that split cannot be reproduced exactly. Rebuilt from the
committed hybrid run, by where the gold page sits:

| gold page within | queries | refused | graded 1 | graded 0 | partial |
|---|---|---|---|---|---|
| top ten pages | 121 | 54 | 34 | 15 | 18 |
| top five pages | 112 | 53 | 28 | 13 | 18 |
| top four pages | 108 | 50 | 27 | 13 | 18 |

The 07-30 percentages sit close to the top-ten row with the partial grades
counted as neither refused nor wrong. Every gen150 query carries one expected
fact, so a partial grade is off the legal grid (the 2026-09-17 caveat), and
`answer_outcome` counts those as wrong. Refusal therefore outnumbers wrong
answers by between about 1.6 and 4 to one, depending on the depth and on where
the partial grades fall. A stronger judge re-grading the 18 partial answers
narrows the range. The ordering, refusal ahead of misreading, and the no-ship
call on the prompt variants stand.
