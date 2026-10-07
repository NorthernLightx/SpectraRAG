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
    --image=<digest printed by the build> --memory=16Gi --cpu=4 \
    --execution-environment=gen2
```

Two constraints the hard way: the build must run in the EU pool
(`--region=europe-west1`) or the multi-GB push to the EU registry times out
from the default US pool, and the deploy should use the image *digest*, not the
`:visual` tag, to dodge stale registry manifests.

## Amendment (2026-09-27): the image carried CUDA torch

The Context above says the image ships the CPU torch wheel. It shipped the CUDA
build: PyPI's Linux torch wheel is built against CUDA and pulls 15 `nvidia-*`
wheels and triton, 3.2 GB compressed, into an image that runs on CPU. The
Python environment was a 4.5 GB compressed layer of the 11.7 GB image.
`pyproject.toml` now resolves Linux torch and torchvision from the PyTorch CPU
index at the versions the image ran (2.10.0 and 0.25.0). The environment
installs to 1.8 GB with no `nvidia-*` package, and the image published from
`main` went from 11.7 to 7.7 GB compressed.

One cold start of the visual image, from its logs: 16 s to import the app, 61 s
to the first model load (importing sentence-transformers, which imports torch),
55 s to load the models and open the index, and 2.5 minutes for the warm-up
query, most of it the first forward passes. Cloud Run streams image bytes as
they are first read, so these phases track how much of the image startup
touches.

Seven starts of the CUDA image took 240 to 283 s, 44 to 61 s of it importing
torch. The first two starts of the CPU image took 245 and 180 s, 37 and 28 s of
it importing torch. The import is shorter in both; two starts are too few to
size the change in the total.

## Amendment (2026-10-07): what moved the cold start and what did not

Two cold starts in early October took 3.5 and 4.4 minutes to become ready, most
of it the warm-up query: 54 to 66 s for the text leg and 122 to 153 s for the
visual leg, against 5 s for the same query once warm. Both checkpoints are
stored in fp32 (the ColQwen2 base at 8.8 GB, bge-m3 at 2.3 GB) and load
memory-mapped, so the first forward passes read the weights from the streamed
image a page at a time.

The service now runs Cloud Run's second-generation execution environment, a
full Linux kernel in place of gVisor's emulated system calls, at the same
price. Two more changes were measured on top of it. Time from the instance
start to the passing startup probe:

| image | after a deploy | on a request after idle |
|---|---|---|
| second generation | 2.5 min | 2.1 min |
| plus a background read of the weights at start | 2.1 min | 2.0 and 2.0 min |
| plus the ColQwen2 base stored in bf16 | 1.7 min | 2.0 min |

The background read cut the warm-up query to 12 s, but the time until the app
imported rose from 14 to 33 s, with the read and Python's imports pulling from
the same image. The total did not move, and the read was dropped.

Every tensor of the ColQwen2 base except one small projection holds values
bf16 represents exactly, and the CPU path loads the model in fp32, so the image
now stores those tensors in bf16. The visual leg's top ten for four probe
queries matched the fp32 image page for page, with scores within 7e-6, the same
on two instances. The compressed image shrank only from 7.84 to 7.34 GB, since
gzip had already squeezed the zero low halves of the fp32 values, and a start
after idle still took 2.0 minutes. Uncompressed bytes are not what bounds the
start.

Where those 2 minutes go, from the per-stage warm-up log: 16 s until the app
imports, 50 s for imports, the text index and bge-m3, 39 s for ColQwen2 and the
page index, and 14 s for the warm-up query, 8 s of it loading the cross-encoder
and reranking.

Max instances is a soft limit: during the 4.4-minute start Cloud Run started a
second instance and stopped the first as it became ready, which added a minute
and paid for two starts. A shorter start leaves less time for that.
