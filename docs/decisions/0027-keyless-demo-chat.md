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
worst possible first impression — the system looks unchattable exactly when a
recruiter or reviewer drops by without a key. Retrieval was already free and
server-side; the missing piece was a generation path with an acceptable abuse
story.

## Decision

A second OpenRouter key, used only by `/demo/chat`, caged in three
independent layers:

1. **Server-chosen free models.** The route ignores any client-sent model and
   walks a fallback chain of `:free` ids (`RAG_DEMO_MODELS`, non-`:free` ids
   dropped at request time). Chain at time of writing:
   `google/gemma-4-26b-a4b-it:free` → `nvidia/nemotron-nano-12b-v2-vl:free`,
   picked by live availability testing (the on-paper-best free model, Kimi
   K2.6, failed 4/4 attempts; these two passed 4/4 at ~1s). Both are
   vision-capable, so the demo path keeps the multimodal showcase (page
   images attached).
2. **Price pinned at the router.** Every upstream request sets
   `provider.max_price = {prompt: 0, completion: 0, request: 0, image: 0}`,
   which OpenRouter enforces strictly: a request that would cost money fails
   with 404 rather than routing to a paid endpoint. Verified live against
   both a free model (passes) and a paid model (blocked).
3. **Credit-limited key.** The key itself carries a $1 limit on the provider
   side, bounding the blast radius of a leak. It reaches the container as a
   Cloud Run secret (`RAG_DEMO_OPENROUTER_KEY`), never the public image.

Quota protection, since the real abuse cost is OpenRouter's account-wide
1000/day free-model quota (shared with local eval work):

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
- Demo answer quality is mid-tier free-model quality by design. The model
  chain is env-overridable because free-endpoint availability churns weekly;
  if the demo starts erroring, re-point `RAG_DEMO_MODELS` before suspecting
  the code.
- Failed upstream attempts still count against OpenRouter's daily free quota,
  so the fallback chain should stay short and exclude known-dead models.
- The OpenRouter account balance must stay positive: a negative balance 402s
  even `:free` requests and silently kills the demo.
