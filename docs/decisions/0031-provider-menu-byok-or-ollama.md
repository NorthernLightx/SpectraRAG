# ADR 0031: Provider menu (generation on the visitor's own provider)

Status: accepted (supersedes [0027](./0027-keyless-demo-chat.md))

## Context

ADR 0027's keyless chat relied on a caged server-side OpenRouter key, and
that model aged badly. The dedicated key was invalidated upstream (401 on
every model), and the `:free` fallback chain it depended on churns. Free
slugs get rate-limited or withdrawn without notice. A server-held key is
also standing operational surface for a portfolio project: quota accounting,
abuse caging, rotation. Meanwhile anyone who wants generation already has a
provider of their own: an OpenRouter key, or a local Ollama.

## Decision

Generation is browser-direct on a provider the visitor picks in the model
menu:

- **OpenRouter (BYOK)**: unchanged mechanics; the key stays in localStorage
  and never touches the server. The model list is fetched live from
  `/api/v1/models` and filtered to vision-capable entries
  (`architecture.input_modalities` contains `"image"`), with a curated
  shortlist pinned on top and substring search over the rest.
- **Local Ollama**: models listed live from `GET /api/tags`, filtered to
  entries whose `capabilities` include `"vision"`. Calls go straight from the
  browser to `localhost:11434` (Ollama's default CORS allows localhost
  origins). Chat uses the **native** `/api/chat` endpoint, not the OpenAI
  compat one: `/v1` cannot set `num_ctx` and loads models at the runtime
  default (4096), which rejects any page-image prompt (~16k tokens). The
  native route takes `options.num_ctx` per request (32k).

Both lists are vision-only because the corpus is text+figures and generation
attaches page images; a text-only pick would 400 or silently answer without
the figures. Provider and per-provider model choice persist in localStorage.

Removed with the caged key: `POST /demo/chat`, the `RAG_DEMO_*` settings,
the `demo_available` health flag, and every frontend demo-key path.

## Consequences

- No keyless generation. Retrieval stays keyless; without a provider the chat
  stops at the retrieved chunks with a how-to notice.
- Agentic search (DCI) still requires an OpenRouter key on either provider.
  The server-side agent spends the caller's key.
- On the Ollama path the OpenAI-style interleaved text/image content is
  flattened to native `messages[].images`; label→image binding relies on
  matching order rather than adjacency, which small local models handle less
  reliably than the big hosted ones.
- The Ollama option only applies where the browser can reach a local Ollama:
  your own machine, not the hosted page.

## Amendment (2026-09-23): where the key lives and where it goes

The key is kept in `sessionStorage`, so it lasts for the tab. "Remember on
this device" in the key menu moves it to `localStorage`, and "Forget key"
clears both. A key left in `localStorage` outlives the visit on a shared
machine, and any script that runs on the page can read it.

"Never touches the server" held for chat but not for agentic search:
`/query/dci` receives the key in the `X-OpenRouter-Key` header and spends it
on that request. The server does not store or log it. Sentry events filter that
header, which the SDK's default header filter does not cover, and carry no
local variables, since the route and the OpenRouter client hold the key in
locals. The key menu says so and asks for a key with a credit limit.

The production build (`web-build/build.mjs`) gives the page a
Content-Security-Policy: scripts other than the listed ones do not run, and
fetches and image loads reach only the listed origins. It does not stop a script
that does run from navigating away with the key, so it narrows the risk rather
than removing it. Scripts are allowed by exact URL, because a CDN host such as
unpkg.com would admit any package published there. Connections are limited to
the page's own origin, the API origin passed with `--api-base`, OpenRouter, and
a local Ollama unless `--hosted`. The no-build dev page has no policy.

## Amendment (2026-10-06): free answers on the hosted page

The hosted page answers questions without a key again, through Google rather
than a server-held key. A visitor with no key gets answers from Gemini
(`gemini-3.5-flash-lite`) through Firebase AI Logic, on the Agent Platform
backend in the `eu` multi-region. A visitor's own OpenRouter key takes
precedence, and the menu works as before.

What differs from ADR 0027's caged key:

- There is no secret on the server or in the page. The page carries the
  Firebase web config, which names the project and grants nothing by itself.
  App Check (score-based reCAPTCHA Enterprise) vouches that a request comes
  from the hosted page, and AI Logic refuses requests without a valid token.
- Google enforces the ceiling, not this code. A service-level spend cap on AI
  Logic pauses it for the rest of the month once spend reaches the budget.
  Enforcement trails usage reporting by minutes, so a burst can overshoot it.
  A paused service answers 403 or 429 and a refused App Check token 401. The
  chat then falls back to the key notice with the search results still shown.
- Each browser gets 20 free answers a day, counted in `localStorage`. The
  count keeps one visitor from spending the month's budget for everyone; it
  is not a security boundary, since App Check and the cap are.
- The reader's messages still come from `/context` (ADR 0033). The browser
  only converts them to Gemini's message format.
- The Firebase SDK and reCAPTCHA load on the first keyless question, not with
  the page.

Gemini 3.5 Flash-Lite was picked over 3.1 Flash-Lite on the four saved demo
questions: both answered the text questions, and only 3.5 read the answer off
a slide chart. That is a sanity check, not an eval.

Given `--firebase-config`, the build bundles the reader and adds AI Logic, App
Check and reCAPTCHA to the policy. reCAPTCHA's script and frame are admitted by
path prefix, because their URLs change with each reCAPTCHA release.
