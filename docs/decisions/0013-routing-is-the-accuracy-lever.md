# ADR 0013 — Routing is the accuracy lever; LLM-classifier router shipped

**Status:** Accepted and applied (2026-05-17). Measured on the correct
MMLongBench corpus, sanity-gate verified.
**Date:** 2026-05-17.

## Context

ADR 0012 rejected the reranker swap and pointed at routing as the real
accuracy lever, but its MMLongBench arm ran against the wrong corpus
(corrected there). This ADR re-ran the question correctly: on the 20
MMLongBench documents the committed `baseline-mmlongbench.json` uses
(`data/mmlongbench/documents/`, NOT the v3 arXiv papers), page-scored via
`scripts/rescore_mmlb_pages.rescore` (MMLongBench labels are page-level).

A hard sanity gate guarded the earlier mistake class: text-only had to
reproduce the committed text-only baseline or abort. It did — recall@10
**0.6721** vs committed **0.685**, nDCG@5 0.5770 vs 0.590. Corpus and
scoring verified correct; the rest is trustworthy.

## Findings (measured, page-level, n=107 in-corpus)

| policy | recall@10 | nDCG@5 | MRR | figure recall@10 |
|---|---|---|---|---|
| text-only | 0.6721 | 0.5770 | 0.5531 | 0.6356 |
| regex-router (was shipped) | 0.7407 | 0.6078 | 0.5852 | 0.7267 |
| **llm-router (gemma3:4b)** | **0.8209** | **0.7426** | **0.7197** | **0.8178** |
| llm-router (qwen3-vl:235b-cloud) | 0.8107 | 0.7393 | 0.7209 | 0.8133 |
| oracle-router (ceiling) | 0.8414 | 0.7609 | 0.7324 | 0.8404 |

1. **Routing is a large, real accuracy lever.** Perfect routing is
   +25.2 % recall@10 over text-only on the non-saturated benchmark.

2. **The repo's prior visual claim is independently validated.**
   `docs/results.md` claimed the regex router lifts recall@10 +9.6 %;
   measured here +10.2 %. Unlike the falsified ~5.5 s latency number,
   this one holds.

3. **The LLM classifier captures ~80 % of the headroom the regex router
   left.** llm-router recall@10 0.8209 = **+22.1 % vs text-only,
   +10.8 % vs the regex router that was shipped**, landing at 0.821 of
   the 0.841 oracle ceiling (regex 0.741 → llm 0.821 → oracle 0.841).
   nDCG@5 gains more: 0.608 → 0.743 (near the 0.761 ceiling). This was
   run with a local `gemma3:4b` classifier — no OpenRouter — so it is
   directly deployable under the project's Ollama-only constraint. It
   confirms the keyless probe (`scripts/experiments/probe_routing.py`: gemma3:4b
   misroutes ~20 % of need-visual queries vs the regex classifier's
   ~78 %).

## Decision (applied)

**The LLM-classifier router is the configuration; it is now wired for
keyless (Ollama-only) deploys.** `src/api/bootstrap.py`
`_build_classifier_from_settings` previously built the LLM classifier
*only* when an OpenRouter key was set and otherwise degraded to the regex
classifier — so under the Ollama-only constraint the API was running the
weak 0.741 regex router. It now falls back to a local Ollama classifier
(`Settings.classifier_ollama_model`, default `gemma3:4b`) instead of the
regex. OpenRouter behaviour is unchanged when a key is present. Change is
two files, mypy-strict + ruff clean, reversible.

Speed cost is bounded and small: routing adds one classify call (~1 s);
the visual MaxSim itself measured **~0.16 s/query** — cheap. Cascade
(ADR 0010) can skip the visual leg on confident-text queries to protect
latency further.

## What this leaves open

- **API text leg has no reranker (separate, pre-existing).**
  `bootstrap.py` builds `PipelineRetriever` without a `reranker=`, while
  the study (and `eval_run`) rerank. The routing improvement is
  orthogonal to reranking — it changes *which leg* fires, and the visual
  leg is the figure/table win regardless — so the *direction* holds, but
  the API's absolute numbers will differ from the study's reranked-text
  measurement. This discrepancy is flagged, **not** silently changed here
  (out of scope; deserves its own ADR).
- **Measurement is retrieval-only (page recall@10/nDCG@5).** That is the
  decisive routing signal and is generator-independent; end-answer
  quality follows retrieval (the established repo pattern) but was not
  separately re-judged in this study.
- **Text-leg rerank is slow on long real-world documents** (~33 s on a
  sampled MMLongBench query vs ~1 s on short arXiv chunks, ADR 0012). A
  separate "good speed" concern (chunk length / `rerank_input_size` on
  long docs), independent of routing.
- **A bigger classifier does NOT close the last ~0.02 to oracle — tested
  and falsified.** The strongest available Ollama model
  (`qwen3-vl:235b-cloud`) scored **0.8107**, *below* `gemma3:4b`'s
  0.8209. Mechanism (evidence: cached decisions): `gemma3:4b` routes
  104/149 to the visual leg (it over-calls `multi_hop`), the cloud model
  only 93/149 (more literal — more `factual`→text). On MMLongBench ~93 %
  of answerable queries are figure/table, and a query sent to the text
  leg is usually a miss, so the cost is asymmetric: **aggressive
  bias-to-visual beats accurate classification.** `gemma3:4b` wins by
  being trigger-happy, not smart. The shipped `gemma3:4b` default is
  therefore validated against the cloud alternative. The remaining ~20 %
  routing headroom is a *policy* lever (explicitly bias routing toward
  visual / tune the prompt), **not** a model-size lever — confirmed by
  measurement, not assumed.
- **The v2 "evidence-location" prompt revealed the routing lever is
  really the *visual leg*, and that the best MMLongBench number is a
  benchmark artifact.** `classify_query_v2` (route by where the answer
  lives, bias-visual-on-ambiguity) drove `gemma3:4b` to **0.846**
  recall@10 — best of all, *above* the 0.841 category-oracle. But the
  cached decisions show why: it routes **107/109 in-corpus queries to
  visual** — it has degenerated to "always-visual", not smart routing.
  It beats the category-oracle only because the oracle trusts golden
  *labels* (some `factual`-labelled queries have visual answers).
  `cloud`+v2 stayed discriminating (93/109 visual) and scored 0.818 —
  *lower here precisely because this benchmark rewards the lazy
  strategy*. MMLongBench is ~93 % visual; the production arXiv corpus is
  text-heavy, where ADR 0007 / `results.md` document the visual leg
  **losing ~10 % on definitional/text queries**. So "always-visual"
  overfits this benchmark and would regress production. **Decision: v2 is
  NOT promoted; `v1` + `gemma3:4b` (balanced, validated) stays the
  shipped default. `classify_query_v2.yaml` is retained as a
  corpus-specific artifact, not shipped, pending a text-corpus eval.**
  The precise lesson: not "a smarter classifier is the lever" but "the
  visual leg is the lever; optimal routing *here* is trivially lazy, and
  the highest single-benchmark number is not the right thing to ship."

## Related

- ADR 0012: rejected the reranker swap, pointed here.
- ADR 0008 / 0007: the routing + hybrid-fusion machinery this exploits.
- ADR 0010: cascade — the lever to keep routing speed-neutral.
- ADR 0004: the visual leg whose value this quantifies (+25 % ceiling).
