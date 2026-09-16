# ADR 0028: Persisted ColQwen2 page index for CPU serving

**Status:** Accepted. The deployed demo serves a real visual-routing leg on a
CPU-only Cloud Run box by loading a pre-built ColQwen2 page index from Qdrant
instead of embedding pages at startup. The offline encode runs once on a GPU;
the server only embeds the query at request time.
**Date:** 2026-06-18.

## Context

The visual leg never built in production. `build_visual_retriever` embeds every
corpus page in-process at startup (`src/rag/retrievers/visual.py`), which needs
a GPU; on the CPU deploy that encode runs for tens of minutes and overruns the
Cloud Run startup window, so `_build_visual_retriever_from_settings` returned
`None` and the app wired a text-only `PipelineRetriever`. `/health` reported
`routing_available: false` and the UI said "router off on this CPU-only
deployment." Figure questions were answered only when text retrieval happened to
land on the right page, whose image the browser then attached at generation.

Offline, the text+visual router is worth **+34.6% recall@10** over text-only on
MMLongBench (0.5545 → 0.7461, `docs/results.md`). The goal is to recover that on
the deployed demo without paying for a 24/7 GPU.

The blocker was never the math or a missing capability. `torch` (CPU wheel) and
`colpali-engine` already ship in the image, and qdrant-client 1.17 supports
multivector collections in embedded `path:` mode. The blocker was that the only
code path embedded pages at startup and kept them in memory, with no persistence
(the `visual.py` module docstring flagged this as unbuilt).

## Decision

Split the page encode (offline, GPU, once) from serving (query-only, CPU), and
persist the page multivectors in Qdrant.

1. **`QdrantVisualStore`** (`src/rag/visual_store.py`) is a multivector
   collection (`size=128`, `MultiVectorConfig(comparator=MAX_SIM)`, base distance
   `DOT`) holding one point per page. DOT, not COSINE: colpali's
   `score_multi_vector` scores raw dot products with no normalization, so COSINE
   would diverge from the eval's ranking on non-unit vectors. Kept separate from
   `QdrantVectorStore`: that store is
   chunk-granular, single-vector, 1024-dim text; this one is page-granular,
   multivector, 128-dim. Both live in the same embedded `qdrant_local`
   directory, so the existing image bake (`COPY qdrant_local`) ships both. New
   setting `visual_collection` (`RAG_VISUAL_COLLECTION`, default
   `rag_corpus_visual`).
2. **Offline build** (`scripts/build_visual_index.py`): renders pages, encodes
   them by reusing `build_visual_retriever` (the same path the eval scores, so
   the persisted vectors are identical to the eval's in-memory ones), and
   upserts the multivectors. Idempotent; `--force` drops only the visual
   collection, not the shared embedded store.
3. **Serve loads, never encodes pages.** `VisualRetriever` takes an optional
   `store`; when set, `retrieve` embeds the query and scores via the store's
   MaxSim, skipping the in-memory path. `_build_visual_retriever_from_settings`
   builds the store, returns `None` if it is empty, and otherwise loads the
   model for query encoding only. The in-memory path stays for eval.
4. **Bake the encoder.** The Dockerfile pre-downloads `vidore/colqwen2-v1.0`
   (adapter + Qwen2-VL-2B base + processor) so startup pays no HuggingFace
   fetch.
5. **Turn it on and size for it.** Deploy sets `RAG_ENABLE_MULTIMODAL=true` and
   bumps Cloud Run to 16Gi / 4 vCPU: the in-process query encoder is ~8GB at
   fp32 alongside bge-m3, and Cloud Run requires ≥4 vCPU at 16Gi.
6. **Classifier fallback.** With multimodal on, the router builds an LLM
   classifier that falls back to an Ollama client when no OpenRouter key is set
   (ADR 0013). Cloud Run has no Ollama, so `classify` would raise and 500 every
   query. `RoutingRetriever.retrieve` now catches a classifier failure and falls
   back to the regex classifier, so a missing or unreachable classifier backend
   cannot take down retrieval.

## Consequences

- The deployed demo serves the real visual router rather than text-only;
  figure-bound pages that text retrieval misses are now retrievable.
- The image grows ~4GB (encoder weights). The visual leg stays inert unless
  `RAG_ENABLE_MULTIMODAL` is set and the visual collection is populated, so the
  bake is harmless on a text-only deploy.
- 16Gi / 4 vCPU costs more per warm second, but `min-instances=0` means that is
  paid only while serving. Each query adds a CPU encode of the 2B model (a few
  seconds at demo QPS); cold starts pay the model load once, which the keep-warm
  `/health` ping hides.
- Retrieval quality tracks the offline number because the persisted vectors come
  from the same encode the eval used and Qdrant's DOT MAX_SIM reproduces colpali's
  dot-product `score_multi_vector`. Verified on a 99-page sample: the Qdrant
  ranking matched a hand-computed fp32 MaxSim reference 6/6, where COSINE matched
  only 4/6 (which is why the metric is DOT). The remaining difference from the
  eval is bf16 (GPU eval) vs fp32 (CPU serve) precision: small adjacent swaps of
  near-tied pages that preserve top-10 set membership. The eval's own bf16
  ranking matched the fp32 reference on only 2/6, so the two dtypes are a wash,
  not a regression. Still a **gate, not an assumption**: the exact production
  recall@10 must be measured by running the eval through the fp32 Qdrant path
  before `data/eval/baseline.json` is moved.
- Supersedes the "if we ever want to persist these" note in
  `src/rag/retrievers/visual.py`. Relates to ADR 0004 (visual retrieval) and
  ADR 0008 (routing). `_collect_pages_from_dir` in `src/api/bootstrap.py` is no
  longer called in production (the visual leg reads the collection, not the page
  directory); it is retained for its unit tests and the `pages_dir` is still
  used to serve page images at generation.

## Redeploy runbook (added 2026-07-27)

The visual image is an overlay on CI's text-only `:main`: `Dockerfile.overlay`
copies the gitignored `rag_corpus_visual` collection into the CI image, and
`cloudbuild.overlay.yaml` builds/pushes it (both now committed). Deploying the
moving `:main` tag or the CI `deploy.yml` workflow reverts prod to text-only.
The index rides only in the overlay.

```sh
gcloud builds submit --region=europe-west1 --config cloudbuild.overlay.yaml .
gcloud run deploy spectrarag --region=europe-west1 \
    --image=<digest printed by the build> --memory=16Gi --cpu=4
```

Two constraints the hard way: the build must run in the EU pool
(`--region=europe-west1`) or the multi-GB push to the EU registry times out
from the default US pool, and the deploy should use the image *digest*, not the
`:visual` tag, to dodge stale registry manifests.
