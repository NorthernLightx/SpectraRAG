# ADR 0014: Wire the reranker into the API retrieval path

**Status:** Accepted and applied (2026-05-18).
**Date:** 2026-05-18.

## Context

Every eval, golden baseline, and the routing study (ADR 0013) runs
dense + BM25 → RRF → cross-encoder rerank. `src/api/bootstrap.py`
`_wire_retriever_from_settings` built the production `PipelineRetriever`
with **no `reranker=`** (only `candidate_pool=settings.rerank_top_k`), so
the *live* API served unreranked retrieval. The system never delivered
the retrieval quality every ADR measured. The measured wins were not
reaching users. Flagged as a separate open item in ADR 0012/0013; this
closes it.

## Decision

Construct `BgeReranker(length_norm=True)` and pass it to the API
`PipelineRetriever`. This mirrors the **validated baseline config** that
produced every committed number: `bge-reranker-v2-m3` (the `BgeReranker`
default) plus ADR 0009 length normalisation. No new settings knob:
deliberately matching the baseline rather than adding speculative config
(same discipline as ADR 0012's rejected `rerank_model` knob). Two lines +
one import.

Lands in the same wiring as ADR 0013's classifier change, so the live
retrieval path is now the full validated pipeline: dense+BM25+RRF →
rerank → routing (Ollama LLM classifier) → text/visual fusion.

## Validation

`src.api.bootstrap` imports; `ruff` and `mypy --strict` clean on the
changed file. The cross-encoder GPU-coexists with ColQwen2 on the 8 GB
card, proven feasible by the routing study, which ran exactly this
`PipelineRetriever` + `BgeReranker(length_norm=True)` alongside the
ColQwen2 visual leg to completion.

## What this leaves open

- **Reranker latency is corpus-dependent.** ~1 s/query on the short
  arXiv demo corpus (acceptable); ~33 s/query was observed on
  MMLongBench's long real-world documents (10-Ks) in the routing study.
  The deployed demo corpus is short, so this is fine; if a long-document
  corpus is ever served, mitigate via the ADR 0010 cascade (skip
  rerank/visual on confident-text) or a CPU/skip setting. Not solved
  here. Flagged, and corpus-appropriate for the demo.
- Answer quality follows retrieval (established repo pattern); not
  separately re-judged for this wiring change.

## Related

- ADR 0012 / 0013: flagged this discrepancy under "what this leaves open".
- ADR 0009: length-normalisation the reranker config mirrors.
- ADR 0010: cascade, the latency mitigation for long-doc corpora.
- ADR 0008: routing. ADR 0013's classifier change lands in the same path.

## Amendment (2026-09-23): one config for the served and the measured stack

This ADR fixed one drift between the API and the eval: the API ran without the
reranker. The same class of bug came back in ADR 0032's 2026-09-17 amendment,
where the server never read `routing_mode`, and it was still open elsewhere:
the Cloud Run image set its reranker, pool size and embedder through separate
env vars, `bootstrap.py` hard-coded length-norm on, and no committed baseline
matched that combination.

Both entry points now build the retriever from one `RetrievalConfig`
(`src/rag/retrieval_config.py`):

- `RetrievalConfig.from_settings` reads the served stack. `eval_run` builds the
  same object from its flags, or from a named profile with `--profile`.
- `src/config/profiles/cpu.yaml` holds the stack the Cloud Run image and
  `spectrarag serve` run (`RAG_PROFILE=cpu`). A unit test checks the image
  bakes the reranker that profile names.
- Every run JSON records `retrieval_config` and a 12-character
  `retrieval_fingerprint`; `/health` reports the fingerprint of what was wired.
  Equal fingerprints mean the two retrieved the same way.

`spectrarag serve` used to rerank a 50-candidate pool per leg while Cloud Run
used 20. Both now use the profile's 20.
