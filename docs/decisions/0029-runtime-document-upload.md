# ADR 0029: Runtime document upload (flag-gated, text-leg, incremental)

**Status:** Accepted. A `POST /ingest` route lets a local operator add a PDF to
the live corpus at runtime, behind `RAG_ENABLE_UPLOAD` (off by default, so the
public demo is unchanged). The new document is text-retrievable on the next
query with no restart.
**Date:** 2026-06-26

## Context

The corpus was build-time only: `scripts/bootstrap_corpus.py` ingests a folder
of PDFs into Qdrant before the server starts, and the server scrolls that
collection once at startup to build the in-process BM25 index and `chunks_by_id`
map (`src/api/bootstrap.py`). A user who wanted to query their own document had
to clone the repo, run the script, and restart. Every comparable tool lets you
drop in a PDF and ask a question, so this was the single biggest usability gap.

## Decision

Add an opt-in upload route that appends to the *same* in-process retrieval
objects the wired retriever already reads, so the change is visible immediately:

1. **`POST /ingest`** (`src/api/routes/ingest.py`) takes a single PDF
   `UploadFile`, runs it through the existing `ingest_paper` pipeline (Docling
   chunking, bge-m3 embedding, Qdrant upsert, `Bm25Index.add`), then updates the
   shared `chunks_by_id`. `Bm25Index.add` invalidates its cached model, so the
   next `/query` rebuilds BM25 over the larger corpus; the dense leg reads the
   freshly upserted vectors from Qdrant. No restart, no re-scroll.
2. **Flag-gated off by default.** `Settings.enable_upload` (`RAG_ENABLE_UPLOAD`)
   defaults False. The public Cloud Run demo leaves it unset, so the route 403s
   and the baked-corpus, no-abuse posture is unchanged. Local users opt in.
   Enabling it on anything networked should be paired with `RAG_PUBLIC_API_KEY`:
   the route carries no auth or rate limit of its own, and the Docling parse runs
   synchronously, so a crafted PDF could stall a shared single-instance deploy.
3. **Text leg only.** The visual leg needs a GPU-built ColQwen2 page index
   (ADR 0028) that can't be produced on the CPU demo box, so an uploaded
   document is text-retrievable but not part of the visual index. Routing still
   works: text-routed queries surface it; visual-routed queries do not.
4. **Bounds.** PDF-only, 25 MB cap, parse failures return 422 (the bad file is
   rejected and the running corpus is untouched). The `paper_id` is the
   sanitised filename stem.

## Consequences

- A local user can `spectrarag serve`, open the UI, upload a PDF, and query it in
  the same session. That is the "use it on my docs in five minutes" path the
  project lacked, and the gap every alternative closes.
- The append mutates process-local state (BM25 + `chunks_by_id`) and Qdrant. It
  is **not** concurrency-guarded: a query racing an in-flight upload can rebuild
  BM25 mid-append. Acceptable for the single-operator local use this targets, and
  the multi-user demo has the route off. A public multi-user upload would need a
  per-session collection, TTL eviction, and a lock, deliberately out of scope.
- Uploaded vectors persist in the active Qdrant collection (no auto-eviction), so
  a local corpus grows across uploads, which is the intended behaviour.
