# ADR 0027 — Keyless demo chat through a caged server-side OpenRouter key

**Status:** Accepted. The deployed demo answers chat turns without a visitor
key via `POST /demo/chat`: the server proxies OpenRouter with its own
dedicated key, restricted so that abuse cannot spend money — only exhaust a
bounded daily quota. BYOK stays the primary generation path and the upgrade
path for stronger models.
**Date:** 2026-06-09.

## Context

Generation was browser-direct BYOK only: without an OpenRouter key the demo
retrieved chunks but produced no answer. For a portfolio demo that is the
worst possible first impression — the system looks unchattable exactly when
someone drops by without a key. Retrieval was already free and server-side;
the missing piece was a generation path with an acceptable abuse story. The
cage below deliberately does not depend on secrecy: every layer is enforced
upstream of any attacker, so documenting it costs nothing.

## Decision

A second OpenRouter key, used only by `/demo/chat`, caged in three
independent layers:

1. **Server-chosen free models.** The route ignores any client-sent model and
   walks a fallback chain of `:free` ids (`RAG_DEMO_MODELS`, non-`:free` ids
   dropped at request time). The chain is env-overridable and ordered by
   measured availability — free endpoints are individually flaky, so the
   fallback is what carries uptime, not any single model. All chain members
   must be vision-capable: the demo path keeps the multimodal showcase (page
   images attached).
2. **Price pinned at the router.** Every upstream request sets
   `provider.max_price = {prompt: 0, completion: 0, request: 0, image: 0}`,
   which OpenRouter enforces strictly: a request that would cost money fails
   rather than routing to a paid endpoint. Verified live against both a free
   model (passes) and a paid model (blocked).
3. **Credit-limited key.** The key itself carries a small hard credit limit
   on the provider side, bounding the blast radius of a leak. It reaches the
   container as a Cloud Run secret (`RAG_DEMO_OPENROUTER_KEY`), never the
   public image.

Quota protection, since the real abuse cost is the account-wide daily
free-model quota rather than money:

- per-IP `30/hour` via the existing slowapi limiter;
- a global in-process counter (`RAG_DEMO_DAILY_CAP`, default 300/UTC-day)
  kept well under the account ceiling. Single-replica state is fine — the
  deploy runs `max-instances=1` — and a restart resetting the counter errs in
  the harmless direction.

When the cap is hit the route returns 429 `demo_quota_exhausted` and the UI
opens a bring-your-own-key modal; retrieval keeps working regardless.

## Consequences

- A visitor can chat immediately; the key prompt becomes an upgrade
  ("stronger models"), not a gate.
- Demo answer quality is mid-tier free-model quality by design; BYOK is the
  quality path.
- Demo availability now depends on the free-model pool, which churns — hence
  the env-overridable chain rather than hardcoded ids.
