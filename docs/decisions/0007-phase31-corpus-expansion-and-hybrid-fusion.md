# ADR 0007 — Phase 3.1 corpus expansion + golden v3 + offline hybrid re-evaluation

**Status:** Accepted. Hybrid (text + visual) RRF fusion remains rejected as a
default — but the per-subset evidence is strong enough that **per-query
routing is promoted to Phase 3.2 priority**: figure/table-category queries
should route through hybrid, definitional/factual queries should stay
text-only.
**Date:** 2026-05-03.
**Phase:** 3.1.

## Context

ADR 0004 deferred hybrid text+visual fusion as the natural Phase 3.1
follow-up: text wins on definitional precision, visual wins on multi-hop /
term-mismatch coverage, both have complementary failure modes, RRF should
combine them.

The first attempt at offline fusion (`scripts/eval_hybrid.py`, this session,
5 papers / golden v2 / 17 in-corpus queries) produced a null result —
hybrid −5.8% nDCG@5 and −11.4% MRR vs text @ page. Two confounds:

1. **Corpus too small for text to fail.** At 5 papers × ~25 pages, text
   @ page recall@10 saturated at 1.0 mechanically — no headroom for visual
   to contribute coverage. Any deviation by visual was a strict ranking
   loss.
2. **Query mix biased toward text.** Golden v2 had 1 figure query, 1
   equation query, 0 table queries — heavily definitional / factual lookup.
   ColPali-style retrieval has its claimed strength on figure/table-grounded
   content the text path can't index well, but we weren't asking those
   queries.

Phase 3.1 fixes both:

- Expand the corpus 5 → 20 papers.
- Add 16 figure/table-targeted queries (golden v3).
- Re-run text @ page, visual, and hybrid; report aggregate + per-subset.

## Implementation

### Corpus expansion (5 → 20 papers)

- `scripts/fetch_papers.py` extended to hit cs.CV / cs.LG / stat.ML.
  Fetched 30 candidates from late-Apr-2026 ArXiv submissions; deduped 4
  that overlapped the existing 5.
- `scripts/inspect_candidates.py` (new, kept for future expansion) scored
  each candidate by a visual-richness composite:
  `1.5 · pages_with_figures + 2.0 · table_pages + 0.05 · pages` with a
  hard 8–45 page gate. Picked top 15 keepers; rejects deleted.
- All 15 papers parse cleanly through `extract_pages` + `chunk_pages`
  (smoke-tested separately via `scripts.ingest --qdrant :memory:` —
  total 2 436 chunks across 20 papers, no NaN crashes after the logger fix
  below).

### Golden v3 (23 → 39 queries; 16 new)

- `data/golden/v3.yaml` carries forward all 23 v2 queries unchanged so
  the v2-subset metrics are directly comparable, then adds:

| | n | type |
|---|---|---|
| Caption-derivable figure queries | 6 | text-friendly figure queries on 6 of the 15 new papers |
| Image-grounded figure queries | 4 | authored after directly inspecting PNG renderings of Figs 6/7 in 2604.28176v1 + Figs 1/2 in 2604.28182v1 |
| Table-content queries (incl. cell lookup) | 4 | table caption + Janus-Pro-7B rank in AEGIS Table 8 |
| OOC negatives | 2 | unrelated topics on the new papers |

- Every cited `relevant_chunk_id` verified to exist via re-running
  `scripts.dump_chunks` on each paper.

### Visual stack — install fix + model dispatcher

- ADR 0004's visual retriever import (`from colpali_engine ...`) was
  silently broken after the recent `torch 2.6 → 2.11` cu126 upgrade
  (commit 96a4fe7) — `colpali-engine` was not in `pyproject.toml` deps.
- Re-added: `colpali-engine>=0.3.15`. The 0.3.15 release pins `peft<0.19`,
  but `transformers 5.6.2`'s `load_adapter` path imports
  `_maybe_shard_state_dict_for_tp` from `peft.utils.save_and_load` — added
  in peft 0.19. Added a `[tool.uv] override-dependencies = ["peft>=0.19.1"]`
  to force the resolver past colpali's overly-conservative cap; both
  imports verified post-override.
- `src/rag/retrievers/visual.py` now has `_select_col_classes(model_name)`
  dispatcher so swapping between `ColPali`, `ColQwen2`, `ColQwen2_5` is a
  `--model` flag away.

### Visual model choice — pragmatic constraint

- The 2026-defensible upgrade is `vidore/colqwen2.5-v0.2` (Qwen2.5-VL-3B,
  ~6 GB at bf16). Attempted but **OOMs on the 8 GB RTX 3070 dev box** —
  Windows desktop compositor + Ollama runtime hold ~3.3 GB regardless of
  loaded models, leaving ~4.6 GB free.
- Fell back to `vidore/colqwen2-v1.0` (Qwen2-VL-2B, ~4 GB) — same
  architectural family, fits headroom. Default in `eval_visual.py` reverted
  with an annotated comment explaining the constraint.
- Newer ColQwen3 / ColNomic / ColModernVBERT exist but were not tried;
  the 4 GB ceiling effectively locks us out of the 4 B-param tier.

### Logger fix (regression caught en route)

- During the 20-paper smoke ingest, paper 2604.28177v1 (AEGIS) hit the
  documented bge-m3 NaN warning path on a chunk containing CJK fullwidth
  parentheses (`（`). Windows stdout's cp1252 encoder couldn't encode
  the char and the (unconfigured) `scripts/ingest.py` structlog path
  crashed via `PrintLogger`.
- Fix in `configure_logging()`: `sys.stdout.reconfigure(errors="replace")`
  if available. With logging configured (eval scripts already did this)
  Python's `Handler.handleError` swallowed the error but truncated log
  lines; `errors="replace"` lets the full message reach stdout with `?`
  substitution. Regression test in `tests/unit/test_logging.py`.
- `scripts/ingest.py` now also calls `configure_logging()` so it shares
  the hardened path.

## Headline result — three stacks at 20 papers × golden v3

Run ids:
- text @ page: `2d818239dbc0` (re-scored from text run `d8ff80ee9258`)
- visual: `69d92c7cdd97` (`run-visual-20260502-234345.json`)
- hybrid: `6d57de8bbda1` (`run-hybrid-20260502-235300.json`)

### Aggregate (in-corpus n=31)

| Stack | nDCG@5 | recall@10 | MRR |
|---|---|---|---|
| **text @ page** | **0.8628** | 1.0000 | **0.8167** |
| visual (ColQwen2-v1.0) | 0.6780 | 0.9677 | 0.6637 |
| hybrid (RRF) | 0.8226 | 1.0000 | 0.7826 |

Hybrid still loses on aggregate (−4.7% nDCG@5, −4.2% MRR vs text @ page).
Same direction as the 5-paper run; smaller magnitude (was −5.8%/−11.4%).

### Per-subset — the sign flip

| Subset | text@page nDCG@5 | visual nDCG@5 | hybrid nDCG@5 | hybrid Δ |
|---|---|---|---|---|
| v2 subset (n=17, definitional / factual) | 0.8268 | 0.6033 | 0.7393 | **−10.6%** |
| **v3-only (n=14, figure / table / image-grounded)** | 0.9066 | **0.7687** | **0.9236** | **+1.9%** |

Two things in this table:

1. **Visual alone jumped from 0.603 → 0.769 nDCG@5** between subsets.
   ColPali-style retrieval really is more useful on figure/table-grounded
   content. ADR 0004's per-query intuition (q4 / q9 / q11 / q12 / q20)
   reproduces and generalises to the v3 figure/table queries.
2. **Hybrid sign flipped.** On the figure/table subset, RRF over
   text+visual edges text-only by +1.9 % nDCG@5 (and +2.7 % MRR). Modest
   on N=14 but the *direction* is what we predicted from ADR 0004 once the
   query mix tilts away from definitional.

### Implications for routing — concrete

An oracle router that picked `max(text@page, hybrid)` per query would hit
**~0.93 nDCG@5** vs 0.86 text-only or 0.82 hybrid-only. Even an imperfect
classifier (say, 80 % accuracy on category) would clear text-only.

Concrete observations for a heuristic:
- Hybrid wins on `category in {figure, table, multi_hop}` for queries the
  text retriever doesn't already perfect (e.g., q4 +0.27, q9 +0.50,
  q37 +0.37).
- Hybrid loses on `category=factual` definitional queries text already
  nails (q6 / q7 / q8 each go 1.000 → 0.5).
- Text-only at-page-granularity is a strong baseline — we should *not*
  fuse for queries the text path already serves at nDCG@5 = 1.0.

The simplest viable router: detect category via length / lexical
heuristics on the query text, and only invoke the visual path when the
classifier signals figure / table / multi-hop.

## Decision

1. **Reject hybrid (RRF over text + visual) as a default-on configuration.**
   The aggregate is still net-negative on the v3 corpus. Same call as the
   5-paper finding, with stronger evidence (n=31 in-corpus instead of
   n=17).

2. **Accept the 20-paper corpus + golden v3 as the new evaluation
   substrate** for visual-vs-text comparisons going forward. v3 stays
   draft until the user reviews the v3-only queries (q24–q39); the v2
   queries are unchanged.

3. **Promote per-query routing to Phase 3.2 priority.** The empirical case
   was already in ADR 0003 (query-expansion) and ADR 0004 (visual);
   v3-only's +1.9 % closes it: route by query category, not by retriever.

4. **Keep `data/eval/baseline.json` pointing at the v2 chunk-level
   baseline (`7b5242df5b38`) for now.** The new v3 page-level numbers are
   an alternative track, not a replacement; promoting them changes the
   regression-gate semantics and should wait until v3 queries are
   reviewed and we have a Phase 3.2 router number to baseline.

## Caveats & open questions

1. **N is small.** v3-only is 14 in-corpus queries; the +1.9 % aggregate
   moves about 0.7 nDCG-points if a single query flips. Direction is
   robust (1 strong win q37, 1 modest loss q26, 12 ties), magnitude isn't.
2. **Visual is `ColQwen2-v1.0` (Qwen2-VL-2B, late 2024).** ColQwen2.5-v0.2
   / ColQwen3 / ColNomic / ColModernVBERT may close more of the gap, but
   the 8 GB VRAM ceiling on this dev box with display attached makes the
   3 B+ tier infeasible. Future hardware unlocks the comparison.
3. **Query authoring bias.** Most v3-only queries are caption-derivable
   (figure caption text appears in chunks, so text rerank can answer
   them). The truly visual-only queries — figure-internal layout, chart
   colour-coding, table cell relationships not flat-extractable — would
   need a vision tool in the authoring loop. Q37 is the only image-grounded
   query with a clean visual win.
4. **Cross-paper visual similarity at scale.** Q26 (Frechet Table 4
   ImageNet) failed for visual — picked the wrong page in the same paper.
   ADR 0004 flagged this for q13 at 5 papers; at 20 papers the failure
   mode persists. Worse with corpus size.
5. **Hybrid recall@10 = 1.000 on every in-corpus query** — fusion never
   *loses* coverage, only ranking. Promotes the case for routing rather
   than dropping visual entirely.
6. **The page-level metric coarsens granularity.** Text @ page (0.8628)
   beats text @ chunk (0.7214 from the original v2 baseline) by ~12 %
   purely from the metric change, not stack improvement. Worth calling
   out so we don't double-count the win.
7. **No generation / judge in this run.** Retrieval-only — `--rerank`
   without `--generate --judge` for speed. End-to-end faithfulness on the
   v3 mix is open.

## References

- ADR 0003 — query expansion (rejected, per-query wins on multi-hop /
  term-mismatch — same routing lesson).
- ADR 0004 — Phase 3 visual retrieval (accepted complementary; deferred
  hybrid).
- `scripts/eval_hybrid.py` — offline RRF fusion of two existing run JSONs.
- `scripts/inspect_candidates.py` — visual-richness scoring used for the
  corpus expansion.
- `data/golden/v3.yaml` — 39-query golden, draft.
- `data/eval/runs/run-20260502-235237.json` — text run (id `d8ff80ee9258`).
- `data/eval/runs/run-text-page-20260502-235300.json` — text @ page
  re-scored baseline (id `2d818239dbc0`).
- `data/eval/runs/run-visual-20260502-234345.json` — ColQwen2-v1.0 run
  (id `69d92c7cdd97`).
- `data/eval/runs/run-hybrid-20260502-235300.json` and `.compare.md` —
  hybrid (id `6d57de8bbda1`) + side-by-side comparison.
- `pyproject.toml` `[tool.uv].override-dependencies` for the
  colpali-engine / peft conflict.
- `tests/unit/test_logging.py::test_stdout_handler_handles_non_cp1252_chars_without_crashing`
  — regression test for the Windows cp1252 logger crash.
