# ADR 0005 — Phase 4 deploy + observability scaffold

**Status:** Accepted (scaffold). Production rollout gated on a successful
`terraform apply` against a real Azure subscription + a soak run against the
deployed instance. Until then this is a green-on-CI scaffold.
**Date:** 2026-05-01.
**Phase:** 4 (scaffold subset: deploy infra + OTel + Sentry).

## Context

PROJECT.md §5 calls for: Terraform → Azure Container Apps deploy, GitHub
Actions full CI/CD, OpenTelemetry SDK with auto-instrumentation, OTLP
exporter, span hierarchy on `/answer`, OTel metrics for tokens / latency /
errors, Sentry, W3C `traceparent` propagation, field-name-aware PII
redaction, and a rotating-file handler decision. Phase 3 closed 2026-05-01
(see ADR 0004). This ADR covers the scaffold subset the user requested:
**deploy infra + OTel SDK + Sentry**. PII redaction, caching, demo, and a
full `timed_event` → span migration are explicitly deferred.

## Decisions

### 1. SDKs follow the Langfuse no-op-when-unconfigured pattern
`configure_otel()` and `configure_sentry()` mirror `make_langfuse_client`:
read SDK-native env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, `SENTRY_DSN`),
return early when unset. They are NOT in `Settings` — by convention, only
project-prefixed `RAG_*` env vars go through `Settings`; third-party SDK
env vars stay outside. Tests can run without an
OTLP collector or Sentry project. Idempotent so duplicate calls (test
fixtures, FastAPI reload) don't double-register handlers.

### 2. Stdout-only logging, drop the rotating-file question
The Container App writes JSON to stdout; Container Apps' built-in Log
Analytics ingestion picks it up. No rotating file handler in the container.
`logs/api.log` stays as the local-dev sink (FastAPI factory still defaults
to it). 12-factor; deferred to Phase 4.x if a separate handler is needed.

### 3. OTel and structlog coexist; no `timed_event` removal
PROJECT.md §5 says spans "replace flat `*.done` events." In practice they
serve different consumers (grep/jq vs. trace UI) and removing the log
records would break the existing `logs/*.log` analysis workflow used for
local debugging. Spans are added at the seams (`/answer`, retrieve, generate)
as a working demonstration; the structlog records stay. Future migration
is a separate ADR.

### 4. X-Request-ID stays; W3C traceparent is added alongside it
The Phase 1.3 `request_context_middleware` is left untouched. OTel's
FastAPI auto-instrumentation honors inbound `traceparent` and emits one
on outbound httpx calls. The two IDs serve different audiences (humans
grepping logs vs. distributed-trace consumers); echoing both is harmless.

### 5. Terraform state: azurerm backend; one-time bootstrap doc
State lives in an Azure Storage container provisioned manually before
first `terraform init`. Documented in `terraform/README.md`. Avoids the
chicken-and-egg of "Terraform managing its own state backend."

### 6. CD is manual until first green deploy
`.github/workflows/deploy.yml` is `workflow_dispatch` only. Auto-deploy on
push-to-main is added in a follow-up PR after a human-driven dispatch
verifies the pipeline against a real Azure subscription. Avoids the
common "first deploy breaks everything and main is now broken" failure.

### 7. Span hierarchy seeded only on `/answer`
`/answer` is wrapped: parent `POST /answer` → child `retrieve` → child
`generate`. `/query` is left to the FastAPI auto-instrumentation default
(one span per request). New spans for visual retrieval, eval runner, etc.
ship in their own commits.

## Caveats

- ColPali / ColQwen2 path is in-tree but not wired into the deployed app
  (visual is `scripts/eval_visual.py`, separate CLI). The deploy serves
  text-path only — same as `data/eval/baseline.json`.
- Container Apps' minimum scale is 0; expect cold-start of ~5 s on first
  request. Acceptable for a portfolio demo; revisit with `min_replicas=1`
  if latency under cold conditions matters.
- `qdrant-client` does not have OTel auto-instrumentation in the same
  package as fastapi/httpx; outbound Qdrant gRPC traces appear as plain
  httpx spans (Qdrant client uses httpx for its REST mode). Acceptable
  for the scaffold; revisit if gRPC mode is enabled.

## References

- `PROJECT.md` §5.
- `src/observability/langfuse.py` — pattern reference for no-op SDKs.
- `src/observability/logging.py` — `truncate_long_strings()` placeholder.
- ADR 0004 — Phase 3 (closed prerequisite).
