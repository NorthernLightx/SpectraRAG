# Finding the bottleneck

How a measurement campaign overturned this project's own founding assumption,
and what it killed along the way. Numbers live in [`results.md`](./results.md)
and [`research/2026-05-29-agenda/RESULTS.md`](./research/2026-05-29-agenda/RESULTS.md);
this is the reasoning.

## The assumption

The headline result here is a retrieval win: routing figure and table queries
to a visual retriever lifts page recall by 35 %. The natural next step was more
of the same: better fusion, better reranking, better routing. For months the
working theory was that the system was *retrieval-bound*: find the right page
more often and the answers follow.

The first end-to-end measurement seemed to confirm it. An oracle arm (feed the
model the known-gold pages) scored 0.55; the real pipeline scored 0.24. A 0.31
gap, apparently all retrieval's fault.

## The confound

That 0.31 was wrong. The two arms used different models, different page
counts, and different serving paths. Three variables changed at once. Rerun
with everything held fixed (same 31B model, same prompt, same path, all 149
benchmark queries), the oracle-to-real gap is **0.07**. Retrieval was costing
seven points, not thirty-one.

The decomposition made it concrete. On the answerable set, retrieval puts a
gold page in front of the model **72 % of the time**, and the model converts
only **46-48 %** of those into a correct answer. Of all end-to-end failures,
61-66 % happen *after* the right page was already in context. The system was
generation-bound, and had been the whole time.

## Killing the obvious fixes

Each candidate fix got a measured test before any code shipped. Most died.

**A bigger model.** A 235B model converted retrieved pages no better than the
31B (and on real retrieval, gold page plus four distractors, it was *worse*
in every category; bigger models proved more sensitive to retrieval noise).
A frontier model matched them: three very different readers converged at
~0.46 on oracle pages. Model capacity is not the constraint.

**Fewer pages.** If the model loses answers in a 5-page context, feed fewer
pages? Refuted: accuracy is monotonic in page count (k=1 scores 0.16, k=5
scores 0.36). Distraction cost is negligible; coverage dominates.

**Prompt engineering.** A strict terse-answer, anti-refusal prompt recovered a
couple of formatting artifacts and lost as much elsewhere. Net zero.

**A generation router.** On oracle pages the big model read figures +0.09
better, suggesting "route figure queries to the big model." Gate-tested
before building: on real retrieval the advantage evaporates and the routed
system *loses* to all-31B, 0.34 vs 0.37. The oracle-only advantage was not
realizable. No router was built.

**Agentic query decomposition.** Decomposing questions into sub-queries and
fusing the results *degraded* recall by 0.08 (confidence interval excluding
zero). Benchmark questions are precise; decomposition dilutes exactly the
specificity that finds the gold page. Also not built.

## What survived

Three levers held up under measurement, and they set the roadmap:

- **Feed more when it fits.** On documents small enough for the context
  window, feeding the whole document beats top-5 retrieval by +0.12, and
  fails or loses beyond the window, where retrieval is required. That
  measured crossover became the route-by-fit design
  ([ADR 0024](./decisions/0024-route-by-fit-page-selector.md)): whole
  document when it fits, RAG when it doesn't.
- **Rank the gold page higher.** The oracle's remaining edge over the real
  pipeline is entirely about the gold page being present, worth ~0.06.
- **A better answer extractor.** The benchmark's extract-then-match scoring
  step mis-marks terse correct answers; a stronger extractor is worth ~0.04,
  and auditing the scorer showed the corrected oracle ceiling is ~0.52-0.56,
  about 0.11 above the strict reading.

## What it cost and what it bought

The campaign was a few days of compute on free-tier models with controlled
arms, plus a failure taxonomy over every post-retrieval miss. It produced no
new feature. It killed four attractive projects (bigger generator, context
trimming, generation router, agentic decomposition), each of which would have
consumed weeks and moved nothing, and it re-ranked the roadmap around the
levers that measured real.

The habits that made it work are the ones this repo tries to practice
everywhere: hold arms controlled before trusting a gap, gate-test a design on
real conditions before building it, treat judge noise as a first-class error
bar, and audit the scorer before believing the score.
