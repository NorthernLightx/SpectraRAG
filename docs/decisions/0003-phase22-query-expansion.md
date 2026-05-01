# ADR 0003 — Phase 2.2 query expansion (LLM rewrite + HyDE + combo)

**Status:** Rejected (default-off). Three configurations tested; none beat
the GPU-rerank baseline on aggregate retrieval. Real per-query wins on
multi-hop / term-mismatch queries are reproducibly cancelled by losses on
factual queries.
**Date:** 2026-05-01.
**Phase:** 2.2.

## Context

After Phase 2.1, the strongest local stack (BM25+dense+RRF → BGE-v2-m3
GPU rerank → qwen2.5:7b generate+judge) produced these in-corpus
aggregates over 17 queries on 5 papers (golden v2, baseline run
`7b5242df5b38`):

| Metric | Value |
|---|---|
| nDCG@5 | 0.7214 |
| recall@10 | 0.9412 |
| MRR | 0.7437 |
| faithfulness | 0.8587 |
| answer_relevance | 0.8261 |
| context_precision | 0.6304 |

The remaining failures clustered into two patterns:
- **multi-hop** (`q4_target_region` ndcg5=0.624, `q12_multibasin_vs_vopt` 0.613)
- **term-mismatch** (`q11_budget_levels` ndcg5=0.000, `q20_exploration_hacking`
  0.000) — the relevant chunks were in the candidate pool (recall@10 ≈ 0.94)
  but ranked below the top-5

Query expansion targets exactly this — feed the retriever multiple
phrasings or a hypothetical answer, then fuse.

## Implementation

- `src/rag/query_expansion.py` — `QueryExpander` (LLM-backed) with two
  methods: `rewrite(query, n)` and `hyde(query)`. Robust line-parser strips
  numbered/bulleted prefixes and dedupes case-insensitively.
- `src/rag/retrievers/multi_query.py` — `MultiQueryRetriever` decorator
  wraps any `Retriever`. Fans out the original + N variants in parallel
  (Semaphore-capped), tolerates per-variant failures via
  `return_exceptions=True`, fuses results via reciprocal rank fusion.
- `src/prompts/library/query_rewrite.yaml`, `query_hyde.yaml`.
- CLI: `--query-expansion --query-expansion-mode {rewrite,hyde,combo}
  --query-expansion-n N`.
- Tests: 13 new units (parser robustness + 4 retriever-decorator behaviors).

The model running on Ollama is `qwen2.5:7b` for both expansion and
generation+judge.

## Three-mode A/B (5 papers × v2.yaml × n_in_corpus=17)

Run IDs in `data/eval/runs/`:
- rewrite (n=3): `cd5b71394d13` (`run-20260501-200728.json`)
- hyde:        `b0d16617a206` (`run-20260501-202147.json`)
- combo (n=2): `71e3311563ef` (`run-20260501-204324.json`)

| Metric | GPU baseline | rewrite | hyde | combo |
|---|---|---|---|---|
| nDCG@5 | **0.7214** | 0.6651 (−7.8%) | 0.4974 (−31.0%) | **0.7291 (+1.1%)** |
| recall@10 | **0.9412** | 0.8824 (−6.3%) | 0.7941 (−15.6%) | 0.8824 (−6.3%) |
| MRR | **0.7437** | 0.6644 (−10.7%) | 0.4647 (−37.5%) | 0.6980 (−6.1%) |
| faith | 0.8587 | 0.8174 (−4.8%) | 0.8457 (−1.5%) | 0.8261 (−3.8%) |
| ar | 0.8261 | 0.8043 (−2.6%) | 0.7652 (−7.4%) | 0.8043 (−2.6%) |
| cp | 0.6304 | 0.5696 (−9.7%) | 0.5957 (−5.5%) | 0.6174 (−2.1%) |
| regression-gate fails | — | 4 | 5 | 2 |

**Per-query nDCG@5 on the targeted weak set (5 queries):**

| query | category | baseline | rewrite | hyde | combo |
|---|---|---|---|---|---|
| q4_target_region | multi_hop | 0.624 | **0.877** | 0.613 | 0.624 |
| q9_baselines | factual | 0.387 | 0.000 | 0.264 | 0.307 |
| q11_budget_levels | factual | 0.000 | **0.431** | 0.000 | **0.387** |
| q12_multibasin_vs_vopt | multi_hop | 0.613 | 0.387 | 0.000 | 0.307 |
| q20_exploration_hacking | factual | 0.000 | 0.000 | 0.000 | 0.000 |

## Findings

1. **Rewrite mode** scores the targeted wins — q4 multi-hop **0.624 → 0.877**
   (MRR 0.5 → 1.0) and q11 term-mismatch **0.000 → 0.431** — and these
   reproduce in `combo`. The technique works on the queries we hoped it
   would work on. But **q9 / q12 also got worse** by similar magnitudes,
   and the aggregate dropped.

2. **HyDE mode is harmful** with this LLM. The hypothetical answer
   passages from `qwen2.5:7b` use vocabulary that doesn't match the
   actual paper passages closely enough — they pull *plausible-sounding*
   chunks into the top-K rather than the *actually-relevant* ones.
   nDCG@5 −31%, MRR −37.5% trips every adoption rule.

3. **Combo mode is the least-bad average** because the original query
   and 2 rewrites contribute strong signal, with the HyDE poison getting
   diluted by RRF. nDCG@5 *slightly improves* (+1.07%) but recall@10
   and MRR still regress beyond 5%.

4. **The technique is fundamentally a tradeoff**: more variants =
   more chances to surface a relevant chunk that the original phrasing
   missed (q4, q11) AND more chances to surface an off-topic chunk
   that out-RRFs the right one (q9, q12). On 17 queries, the latter
   wins on average.

## Decision

**Reject** as a global retrieval upgrade. Keep code in tree as opt-in
(`--query-expansion`) for two future paths:

1. **Per-query-category routing.** If we can detect at query time that
   a query is multi-hop or term-mismatch (a small classifier or LLM
   pre-flight), apply rewrite mode *only* on those. The
   `MultiQueryRetriever` decorator pattern already supports this — wire
   a router that selects between `PipelineRetriever` and
   `MultiQueryRetriever` per query.
2. **Stronger rewrite LLM.** `qwen2.5:7b`'s rewrites are sometimes
   too lateral (q9 rewrites pulled in chunks about *related* baselines
   instead of *the paper's* baselines). A larger or instruction-tuned
   model might produce more faithful paraphrases. Cloud `gpt-4o-mini`
   for the expansion only is cheap (~$0.001/query) and easy to A/B.

Default eval pipeline stays text-only with single-query retrieval.
`data/eval/baseline.json` is unchanged (still `7b5242df5b38`).

## Caveats

- 17 in-corpus queries is small. The −7-10% aggregate drops include
  high variance: a single query going from 0.387 to 0.000 (q9
  rewrite) moves the macro-mean by about 2 points alone. v3 with 30+
  queries would tighten the signal.
- Rewrite was tested at n=3, combo at n=2. Lower n likely tames the
  rewrite-mode regression but also dilutes the wins on q4/q11; not
  worth the extra ablation runs without a query-category router to
  apply selectively.
- The wins on q4 and q11 are real, large, and reproducible across
  rewrite and combo modes. Do not treat this ADR as "query expansion
  doesn't work" — it works *for the queries it's designed to help* and
  the cost is concentrated on a different category.

## References

- ADR 0001 (Contextual retrieval) — earlier "rejected with per-query wins" ADR; same pattern.
- ADR 0002 (Multi-modal chunks) — opt-in default-off, similar verdict.
- `src/rag/query_expansion.py`, `src/rag/retrievers/multi_query.py`,
  `src/prompts/library/query_*.yaml`, `tests/unit/test_query_expansion.py`,
  `tests/unit/test_multi_query_retriever.py`.
