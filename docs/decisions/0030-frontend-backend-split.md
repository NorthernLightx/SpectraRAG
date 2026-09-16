# ADR 0030: Split the frontend off the backend image

Status: accepted

## Context

The SPA (`web/`) was baked into the FastAPI image and served by it. That image
is ~8 GB (torch, docling, the bge-m3 model, page renders, the embedded Qdrant
index), so a one-line change to a static `.jsx` file forced a full image
rebuild + push (roughly an hour), even though the frontend is plain static
files with no build step (Babel runs in the browser).

## Decision

Serve the frontend as its own Cloud Run service, separate from the API.

- **`spectrarag-web`**: a tiny nginx image (`Dockerfile.web`) that serves
  `web/`. Builds in seconds; redeploys in ~minutes.
- **`spectrarag`** (API): unchanged; rebuilds only when code or deps change.
- The SPA reads its API base from `window.SPECTRARAG_API_BASE`
  (`web/app/config.js`), defaulting to same-origin so local dev and the
  combined deploy keep working untouched. The prod API URL is injected at
  build time (`--build-arg API_BASE=…`), not committed.
- The API adds CORS (`CORSMiddleware`, origins from `RAG_CORS_ORIGINS`),
  enabled only when that env var is set, so same-origin deploys add no CORS.

The deploy specifics (URLs, the build/deploy commands, the CORS origin value)
are local-only, like the visual-overlay build; this repo keeps the generic
structure, not the private endpoints.

## Consequences

- Frontend iterations stop touching the ML pipeline. UI deploys in minutes.
- Two services to operate instead of one (the frontend one is trivial/static).
- Cross-origin is the one gotcha: the API now needs CORS for the web origin;
  page images load fine cross-origin via `<img>`.
