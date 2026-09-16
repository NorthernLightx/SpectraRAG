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
