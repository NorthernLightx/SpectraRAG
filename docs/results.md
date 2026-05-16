# Eval results

Detailed numbers behind the headline table in [`README.md`](../README.md).
Methodology lives in [`evals.md`](./evals.md); golden sets and committed
baselines under [`data/golden/`](../data/golden/) and
[`data/eval/`](../data/eval/).

## Ablation summary

Three independent A/B experiments on MMLongBench-Doc, each isolating one
component's contribution. Raw run JSONs are linked from the run-id in
each row; per-category breakdowns and methodology in the sections below.

| Component | Baseline | Improved | Δ rel | Test set | Reference run |
|---|---|---|---|---|---|
| **Retrieval** (recall@10) | 0.6854 | **0.7515** | **+9.6 %** | n=111 in-corpus | router (`cc45831697b6`) vs text-only (`589f7269d617`) |
| **Dispatch** (any→hybrid) | 17.4 % | **71.8 %** | **+54 pp** | n=149 queries | LLM classifier (`gpt-4o-mini`) vs regex (ADR 0008) |
| **Generation** (gold-match) | 0.330 | **0.623** | **+89 %** | n=106, identical context | `qwen3-vl-32b` vision vs `gpt-4o-mini` text |

Each experiment isolates one component; we don't have a single end-to-end
run that turns all three on simultaneously, so the numbers don't compound
multiplicatively. They DO compose architecturally: the dispatch upgrade
fires the visual retriever more often (lifting retrieval coverage), and
the visual retriever surfacing the right page is what unlocks the vision
generator's lift on figure-grounded queries.

## Latency profile

Per-stage timings from [`scripts/legacy/profile_latency.py`](../scripts/legacy/profile_latency.py)
over 6 representative queries against a 2,436-chunk corpus. Local dev
stack (Ryzen 5800X / RTX 3070 / 16 GB; Ollama on CPU; Qdrant in Docker).

| Stage | Backend | median | p95 | Notes |
|---|---|---|---|---|
| **Embed query** | Ollama `bge-m3`, CPU | 151 ms | 170 ms | Stable after warmup; first call ~3 s |
| **Dense search** | Qdrant top-50 | 14 ms | 46 ms | HNSW; effectively constant in N |
| **BM25 search** | `rank_bm25`, in-process | <5 ms | — | Bag-of-words; not measured (legacy schema, ingest fresh to measure) |
| **RRF fusion** | pure Python | <1 ms | — | Rank-list math |
| **Reranker** | BGE-rerank-v2-m3, GPU | ~5,500 ms | — | From v2 baseline; dominates whole-query latency |
| **Generation** | `gpt-4o-mini` via OpenRouter | ~2,000 ms | — | Network + remote |
| **Generation (alt)** | `qwen2.5:7b` via Ollama GPU | ~60 s | — | From v2 baseline |

The retrieval-only path (embed + dense + BM25 + RRF) is sub-200 ms on
this hardware. Reranker is the dominant whole-query cost at ~5.5 s — the
right next thing to optimise (smaller cross-encoder, batching, or skip
the rerank for queries the classifier knows are cheap). Generation
latency is mostly network + provider; the BYOK browser-direct path adds
no hops on our side.

## Why multi-modal? — one concrete example

`mmlb_0008` from MMLongBench-Doc, paper `2310.05634v2`, gold page 8:

> *"In figure 5, what is the color of the line that has no intersection
> with any other line?"*  → expected answer: **red**

| Stack | top-10 retrieved pages | recall@10 |
|---|---|---|
| text-only (`589f7269d617`) | `[25, 23, 24, 94, 16, 35, 4]` | **0.00** |
| router (`cc45831697b6`) | `[25, 5, 23, 12, 24, 8, 94, 16, 15]` | **1.00** |

This is the kind of question that's fundamentally unanswerable from
extracted text — the answer lives in the chart's colour-coding. The text
retriever can't surface page 8 because the relevant signal was never in the
text layer; the visual leg (ColQwen2 multi-vector + late-interaction MaxSim
on the rendered page image) recovers it. 6 more queries with the same
shape are listed in the output of `scripts/legacy/find_visual_wins.py`.

**The honest tradeoff.** Multi-modal retrieval helps when figures encode
information as pixels — chart colours, layout geometry, screenshot content,
image-only diagrams. It helps less when figures encode information as a
text layer the PDF parser can extract — most modern arXiv preprints
serialise even figure-internal labels and captions as selectable text,
which is why golden v3's per-query router showed only +1.9 % on figure
subsets while MMLongBench shows +15.3 % on the same category. ADR 0007 +
[`docs/decisions/0008`](./decisions/0008-phase32-routing.md) explain the
mechanism.

## The generation gap (closed)

When the visual leg surfaces the right page, a text-only generator still
cannot read the page image — it answers *"Not stated in the provided
context."* on `mmlb_0008` and friends.
[`scripts/legacy/experiment_mmlb_gen.py`](../scripts/legacy/experiment_mmlb_gen.py) runs
106 in-corpus MMLongBench queries through `gpt-4o-mini` (text-only) vs
`qwen/qwen3-vl-32b-instruct` (vision) with **identical context** —
gold-evidence page text fed to both, plus the rendered page PNGs as
additional content blocks for the vision model.

| Aggregate (n=106) | text `gpt-4o-mini` | vision `qwen3-vl-32b` | Δ rel |
|---|---|---|---|
| **gold-answer match** | 0.330 | **0.623** | **+89 %** |
| answer_relevance | 0.612 | 0.906 | +48 % |
| faithfulness | 0.597 | 0.609 | +2 % (judge-bug bound) |

Per-category, vision lift scales with how visual the category is:

| category | n | text gold-match | vision gold-match | Δ abs |
|---|---|---|---|---|
| factual | 8 | 0.75 | 0.88 | +0.12 |
| **figure** | **75** | **0.27** | **0.61** | **+0.35** |
| table | 23 | 0.39 | 0.57 | +0.17 |

**Why faithfulness stays flat.** The judge prompt sees only the text
context. When vision answers *"the line is red"* (correct, gold-matched)
and "red" isn't in the page text, the judge flags the claim unsupported.
Programmatic gold-answer match bypasses this judge bias and is the channel
to trust — it's a deterministic substring check against MMLongBench's
expert-annotated gold answers.

Run JSONs at `data/eval/runs/exp_mmlb_gen_full.json` (full) and
`exp_mmlb_gen_smoke.json` (7-query smoke pre-registered the prediction;
5/7 vision wins, 0/7 text wins, 2/7 ties).

## The dispatch gap (closed)

The retrieval section flagged that the regex classifier dispatched only
26 / 149 MMLongBench queries to hybrid where 98 were figure/table-evidenced
— a 75 % under-fire on a corpus where natural-language questions don't say
"Figure X" / "Table N". `src/rag/retrievers/classifier_llm.py` adds an LLM
zero-shot classifier as an alternative to the regex.
[`scripts/legacy/exp_classifier_dispatch.py`](../scripts/legacy/exp_classifier_dispatch.py)
ran both over the same 149 queries:

| | regex (ADR 0008) | LLM (gpt-4o-mini) | Δ abs |
|---|---|---|---|
| any → hybrid | 17.4 % | **71.8 %** | **+54 pp** |
| figure → hybrid (n=76) | 25 % | **87 %** | **+62 pp** |
| table → hybrid (n=24) | 12 % | 50 % | +38 pp |

The LLM over-dispatches some factual queries (0 % → 56 %) but the fail-safe
is bounded: hybrid retrieval on a text-only-needed query just costs more
compute, the answer is still correct.

## Composed pipeline — end-to-end win

Three lifts compose into a single deployed answer:

1. **Visual retriever** (ColQwen2 in `RoutingRetriever`): +9.6 % recall@10
   aggregate, +15.3 % on figure subset.
2. **LLM classifier** (`LLMQueryClassifier`, opt-in via `classifier=` on
   RoutingRetriever): dispatches 87 % of figure queries to hybrid vs the
   regex's 25 %.
3. **Vision generator** (`Generator(pages_dir=...)` with a vision-capable
   model): +89 % gold-match on figure-grounded queries when it actually
   fires.

Each layer's lift was measured in isolation; the dispatch upgrade closes
the bottleneck that was previously suppressing the composed product.

## Production baseline — golden v2

5 papers, 23 queries (17 in-corpus). Stack: BM25 + dense + RRF →
BGE-v2-m3 cross-encoder rerank → qwen2.5:7b generate + judge.

| Metric | Value |
|---|---|
| nDCG@5 (in-corpus macro) | 0.7214 |
| recall@10 (in-corpus macro) | 0.9412 |
| MRR (in-corpus macro) | 0.7437 |
| citation grounding | 1.0000 |
| faithfulness (LLM judge) | 0.8587 |
| answer relevance (LLM judge) | 0.8261 |
| context precision (LLM judge) | 0.6304 |
| p50 whole-query latency | ~73 s |
| p50 rerank stage on GPU | ~5.5 s |

CI regression gate fails the build if any metric drops by > 5 %
(`scripts/check_regression.py`).

## Corpus-expansion follow-up — golden v3

39 queries, 20 papers, retrieval-only:

| Stack | nDCG@5 | recall@10 | MRR |
|---|---|---|---|
| text @ page (chunks → page granularity) | **0.8628** | 1.0000 | **0.8167** |
| visual (ColQwen2-v1.0 only) | 0.6780 | 0.9677 | 0.6637 |
| hybrid (RRF text + visual at page level) | 0.8226 | 1.0000 | 0.7826 |

The split that motivates per-query routing: on the 14 figure/table-targeted
queries (q24–q39 in-corpus), hybrid edges text @ page (+1.9 % nDCG@5); on
the 17 definitional v2 queries, hybrid loses (−10.6 %). Full analysis in
[`docs/decisions/0007-phase31-corpus-expansion-and-hybrid-fusion.md`](./decisions/0007-phase31-corpus-expansion-and-hybrid-fusion.md).

### Per-query router — golden v3

Retrieval-only with `--rerank --router` (run `6447247ef8e7`, ADR 0008):

| Routed category | n | mean nDCG@5 | path |
|---|---|---|---|
| equation | 1 | 1.000 | text-only |
| factual | 13 | 0.712 | text-only |
| **figure** | **11** | **0.876** | **hybrid (RRF page-level)** |
| **table** | **4** | **0.875** | **hybrid (RRF page-level)** |
| multi_hop | 2 | 0.619 | hybrid |
| out_of_corpus | 8 | 0.000 | (correct — no relevant chunks) |
| **Aggregate (in-corpus n=31)** | | **0.7942** | mixed |

The router fires hybrid for `figure`/`table`/`multi_hop` and stays
text-only for `factual`/`definitional`/`equation`, exactly per the ADR
0007 §"Implications" oracle.

## Stress test on MMLongBench-Doc

Golden v3 is too easy to differentiate text vs hybrid generation —
PyMuPDF's text-layer extraction captures even figure-internal labels on
modern arXiv PDFs, so caption text is nearly always sufficient.
[MMLongBench-Doc](https://arxiv.org/abs/2407.01523) is the harder regime:
47-page PDFs with 22.5 % unanswerable queries (refusal-gate friendly),
GPT-4o tops out at 44.9 % F1 — a non-saturated benchmark.

20 docs / 149 queries (76 figure + 24 table + 9 factual + 40 OOC),
page-level scoring. Both runs use BGE-rerank + gpt-4o-mini for generation
and as judge.

| | text-only `589f7269d617` | router `cc45831697b6` | Δ rel |
|---|---|---|---|
| **Retrieval** (n=111 in-corpus) | | | |
| nDCG@5 | 0.5904 | 0.6177 | +4.6 % |
| recall@10 | 0.6854 | **0.7515** | **+9.6 %** |
| MRR | 0.5741 | 0.6009 | +4.7 % |
| **figure subset** (n=75) | | | |
| nDCG@5 | 0.5161 | 0.5565 | +7.8 % |
| recall@10 | 0.6378 | **0.7356** | **+15.3 %** |
| **Generation** (post judge-bug fix, all 149) | | | |
| faithfulness | 0.5990 | 0.6074 | +1.4 % |
| answer_relevance | 0.6812 | 0.6879 | +1.0 % |
| context_precision | 0.4315 | 0.4369 | +1.3 % |

The visual leg's lift on figure-subset recall@10 (+15.3 %) is the clean
win that golden v3 couldn't show.

## Multi-modal regression gate

The MMLongBench router run is committed as
[`data/eval/baseline-mmlongbench.json`](../data/eval/baseline-mmlongbench.json)
so future changes can't silently lose the +9.6 % recall@10 win. The eval
runner stores chunk-level scores against `relevant_chunk_ids` (always 0.0
for MMLongBench, since the golden uses page-level relevance). Page-level
scoring lives in
[`scripts/rescore_mmlb_pages.py`](../scripts/rescore_mmlb_pages.py); the
committed baseline is its output.

To check a candidate run against the gate:

```sh
.venv/Scripts/python.exe -m scripts.rescore_mmlb_pages \
    --run data/eval/runs/run-XXXX.json \
    --golden data/golden/mmlongbench-v1.yaml \
    --output /tmp/candidate-rescored.json
.venv/Scripts/python.exe -m scripts.check_regression \
    --baseline data/eval/baseline-mmlongbench.json \
    --candidate /tmp/candidate-rescored.json \
    --metrics ndcg_at_5 recall_at_10 mrr
```

The gate fails if any retrieval metric regresses > 5 %. Verified locally:
running it with the **text-only** run as candidate fails with
`recall_at_10 delta -9.18 %` — exactly the regression we'd see if a future
change disabled the visual leg. CI doesn't run MMLongBench (it needs
Qdrant + Ollama + 15 min of compute); this is a manual gate, run before
merging any change that touches retrieval.

## Failure modes — when this gets it wrong

A repo without honest failure cases is a tutorial. Five curated examples
from `data/eval/runs/exp_mmlb_gen_*.json` and `exp_classifier_dispatch.json`.

### 1. Generation failure — text can't read chart numerics

> **Q:** *"According to the chart on page 14 how much time was spent with
> family and friends in 2010?"*
> **Gold:** 21 %

- **Text-only (`gpt-4o-mini`):** *"Not stated in the provided context."*
- **Vision (`qwen3-vl-32b`):** *"21 %"* ✓

Right page retrieved both ways; text-only LLM cannot extract numerics
from a clock-diagram visualisation. Vision model reads the 2010 panel
directly. Closed by the vision-generator path.

### 2. Vision hallucinates plausibly when retrieval is weak

> **Q:** *"What range does red color represent in approximate distance
> from the Mississippi River in the chart that tracks the West Nile
> Virus in Europe?"*
> **Gold:** 0–375 miles

- **Text-only:** *"Not stated in the provided context."* (no citations)
- **Vision:** *"0 to 375 miles"* — looks correct, **faithfulness = 0.0** per judge

The vision model produces the right answer but the judge rates
faithfulness 0.0 because the retrieved page text doesn't support the
claim. Color-legend semantics live in the image only — the claim is
grounded in pixels, not the retrieved text, and the faithfulness signal
correctly flags that.

### 3. Correct refusal on out-of-corpus

> **Q:** *"Will it rain on Mars tomorrow?"*

- **System:** *"Not stated in the provided context."*
- **Retrieval:** 10 unrelated chunks (nDCG@5 = 0, recall@10 = 0)
- **Judge:** faithfulness 1.0, answer-relevance 1.0

OOC handling works: zero retrieval signal, system declines rather than
confabulating. The OOC refusal gate (ADR 0006) is what makes this score
correctly on the LLM judge.

### 4. Classifier mis-dispatch — LLM over-routes a factual query

> **Q:** *"For dataset construction, which step takes the most word to
> describe than the others?"*
> **Gold category:** factual (text-answerable)

- **Regex classifier:** routes text-only ✓
- **LLM classifier (`gpt-4o-mini`):** routes hybrid ✗

LLM gets confused by "step-wise" wording and routes to vision
unnecessarily. The fail-safe is bounded — costs extra compute, doesn't
break the answer — but it's the cost of the dispatch lift documented
above.

### 5. Comparative chart reading — vision wins clean

> **Q:** *"Which category has the most increase from 2005 to 2010 for time
> spent on weekends?"*
> **Gold:** *Eating out*

- **Text-only:** *"Not stated in the provided context."*
- **Vision:** *"Eating out, rose from 10 % in 2005 to 17 % in 2010
  (+7 pp)"* ✓

Visual comparison across two time-indexed segments. Text extraction
can't reconstruct the trend; vision reads both years and computes the
delta.
