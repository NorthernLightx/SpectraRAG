# ADR 0026 — Direct Corpus Interaction (DCI): a text-IR method, not a multimodal lever, shipped as an experimental opt-in

**Status:** Accepted. DCI (an LLM agent that greps a raw corpus instead of using
an embedding index — arXiv 2605.05242) was implemented and evaluated. It is **not**
the default retriever and **not** wired into routing: on this repo's multimodal
corpus it is dominated by document-scoping and sits off the reading bottleneck.
It ships as an **experimental opt-in mode** (`/query/dci`, `RAG_ENABLE_DCI`,
default off), key-bound and labelled. Honest-negative-flavoured, like ADR 0012,
0013/0015, 0016, 0018, 0019 — the method works on its own turf, just not as a
lever here.
**Date:** 2026-06-03.

## Context

DCI replaces the fixed embedding + top-k step with an LLM agent that searches the
raw text with terminal-style tools (SEARCH/GREP/READ). The paper reports large
gains on text-IR and multi-hop QA. The question this ADR settles: is DCI a lever
for *this* system (multimodal document RAG), and if not, is it worth keeping at
all?

## What was measured

### On its own turf — BRIGHT-Biology (reasoning text-IR)

Implemented the agent (`src/dci/`) and a BRIGHT harness
(`scripts/experiments/dci_*`). nDCG@10, same corpus/gold/metric as the paper:

- **SEARCH-only, no agent: ~3.** The tool is inert without the model — the agent
  supplies the retrieval intelligence (it reformulates the verbose query into
  focused searches). This is the load-bearing control.
- **qwen3-235b read+grep: ~54 (n=103)** — the agentic-text-IR tier, above BM25
  (~19) and dense (~30), below the published DCI-Lite 60. The gap is their richer
  apparatus (real ripgrep pipes, find/sed, 300-turn runs, context management,
  gpt-5.4-nano with high thinking); gpt-5.4-nano in this harness scored ~53.
- **Full-bash tools (FILTER/COUNT/SCRIPT) did not help.** Clean same-model A/B:
  read+grep 54 vs full-bash 50. The paper's "+12 from full bash" did not
  replicate; read+grep is the cost-optimal config.
- **In-context distillation barely transfers.** Few-shot priming a free 4B model
  with strong-model traces moved it ~+2 (noisy) — which supports the paper's
  choice to *fine-tune* small models rather than few-shot them. Fine-tuning is
  out of scope (data + GPU).

### As a lever for this repo — MMLongBench (multimodal)

- **Retrieval is not the bottleneck** here: post-retrieval reading is (the gold
  page is in front of the reader most of the time, but it converts under half —
  the 2026-05-29 capstone; ADR 0013).
- **DCI's recall recoveries are dominated by document-scoping**, which already
  ships (`PipelineRetriever` filters to `paper_id`); the clean lexical edge over
  within-doc dense was a handful of queries
  (`scripts/experiments/{grep_recovers_misses,dense_control_misses}.py`).
- **About a third of answerable queries are pixel-only** (`answer_in_text.py`) —
  the answer lives in chart/figure pixels, where a grep agent is structurally
  blind. PrismRAG's visual leg is the right tool there, not DCI.

## Decision

1. **Do not** make DCI a default retriever or a routing tier. It optimises the
   part that is not losing on this corpus, it is text-only, and it is a slow
   multi-step LLM loop versus sub-second vector retrieval.
2. **Ship it as an experimental opt-in mode.** `DciRetriever`
   (`src/rag/retrievers/dci.py`) runs the agent over the chunk index and maps its
   ranking to `RetrievalResult`s; exposed at `/query/dci`, gated by
   `RAG_ENABLE_DCI` (default off), with a toggle on the Inspection page. Read+grep
   toolset (the A/B winner). This keeps the work usable and demoable without
   pretending it improves the headline.
3. **Key handling.** The agent runs server-side, so the route needs an OpenRouter
   key — the server's own when configured, else the caller's via the
   `X-OpenRouter-Key` header, used in-memory for the request and never logged or
   stored (read from a header, not the body, so it is absent from the request
   log). This is a deliberate, bounded departure from the keyless-server BYOK
   default (generation still goes browser-direct), scoped to this mode and
   surfaced in the UI.

## Honest caveats

- The harness reproduces DCI's *neighbourhood*, not its exact leaderboard number;
  the tooling gap to the paper is real and named above.
- BRIGHT per-query scores are bimodal and subset means swing (an early n=40 slice
  read 65, the full n=103 run fell to ~50); the full set is the trustworthy
  number.
- The experimental mode is unit-tested and type-checked; treat the live path as
  showcase, not a measured production lever.

## Related

- ADR 0019 — agentic query-decomposition refuted on MMLongBench. DCI is a
  different mechanism (raw-corpus tool interaction, not decomposition over a
  retriever) but the same posture: measure before shipping.
- ADR 0013 — routing is the accuracy lever; DCI does not move it.
- Full experiment record: `docs/research/dci-bright-2026-06-02/` (local note).
