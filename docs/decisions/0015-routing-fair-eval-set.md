# ADR 0015 — A routing-fair evaluation set (`robust-v1`)

**Status:** Built, but the empirical fairness validation **FAILED** —
robust-v1 does not discriminate routing under paper-filtered page-level
scoring. Not yet usable for routing tuning. The result **revises ADR
0013** (see "Validation result").
**Date:** 2026-05-18.

## Context

ADR 0013 established that neither existing eval set can judge a *router*
honestly: v3 (arXiv) is text-heavy and saturated; MMLongBench is ~93 %
visual, so it *rewards* a degenerate "always-visual" policy — the highest
MMLongBench number turned out to be a benchmark artifact, not a real
routing upgrade. Without a fair set we cannot safely tune production
routing (changing it risks optimising for a lopsided benchmark and
regressing the real, text-heavy corpus).

## Decision

Build `data/golden/robust-v1.yaml` by **stratified-sampling the existing
human labels** (no LLM-invented golden), balanced by **true evidence
modality**, via `scripts/experiments/build_robust_golden.py` (seeded, reproducible).
Modality comes from MMLongBench's human `evidence_sources` label and v3's
category. 98 queries:

| bucket | total | from v3 | from MMLongBench |
|---|---|---|---|
| text (answer in prose) | 25 | 16 | 9 |
| figure | 26 | 11 | 15 |
| table | 17 | 4 | 13 |
| mixed (text+visual) | 16 | 0 | 16 |
| ooc (unanswerable) | 14 | 7 | 7 |

Scoring is unified to **page level**; v3 chunk-ids → pages via the `::pN`
convention (same as `scripts/rescore_mmlb_pages`).

## Why this is routing-fair (by construction, not assumption)

Balance **plus the measured per-leg asymmetry** makes lazy policies lose:
the text leg scores ≈0 on visual-evidence queries (ADR 0013) and the
visual leg loses ~10 % on genuine text (ADR 0007). So on this set
"always-visual" forfeits the 25 text queries, "always-text" forfeits the
43 visual queries — only correct per-query routing scores high. The
degenerate policy that scored 0.846 on pure-MMLongBench cannot win here.
Fairness is structural.

## Scoring decision (recorded)

The set spans two corpora, so raw page numbers collide across documents.
Every query carries `paper_id`; the consuming eval **must run
paper-filtered retrieval** (scope to the query's document — the realistic
doc-QA setting, and it eliminates the collision). The current routing
study driver does **not** yet paper-filter; the validation run must
enable it. This is a deliberate, recorded methodological choice.

## What this leaves open

- **The genuine-text pool is exhausted at 25.** That is *all* the
  human-labeled text-evidence questions available across both sets — the
  set is as text-heavy as existing labels allow. It is far more balanced
  than either source (v3 ≈ no hard-visual; MMLongBench ≈ 7 % text) and 25
  is enough to make "always-visual" lose decisively, but authoring more
  text-and-visual-hard cases is a future enhancement (the only path to
  that is human/assisted labeling, deliberately out of scope here — no
  model-invented golden).
- **Empirical fairness check not yet run.** The structural argument
  above is sound, but the honest confirmation is to run text-only vs
  always-visual vs oracle on `robust-v1` and show oracle ≫ both lazy
  policies. That needs ingesting both corpora (≈40 docs) + a
  paper-filtered routing pass (~hours). Recommended as the next step
  *before* this set is used to tune production routing.

## Validation result (2026-05-18) — FAILED, and the deeper finding

`scripts/experiments/validate_robust.py` (keyless: always-text / always-visual /
bucket-oracle; paper-filtered; page-level recall@10):

| policy | overall | text | figure | table | mixed |
|---|---|---|---|---|---|
| always-text | 0.887 | 0.920 | 0.881 | 0.859 | 0.875 |
| always-visual | 0.870 | 0.840 | 0.912 | 0.875 | 0.844 |
| oracle | 0.894 | 0.920 | 0.912 | 0.875 | 0.844 |

Oracle beats the best lazy policy by only **+0.0066** (gate ≥ +0.05) →
**not routing-fair.** The modality balance worked; the *scoring regime*
is the failure. `always-text` scores **0.88 / 0.86 on figure / table** —
the premise "text leg ≈0 on visual evidence" is false once retrieval is
**paper-scoped** and scored at **page** granularity: within the known
document both legs surface the right page ~85–92 % of the time (figure
pages carry captions/surrounding text; page-level recall@10 on short docs
is near-saturated, the same way v3 saturates).

**This revises ADR 0013.** Its "routing is a +25 % lever" was measured
*un-paper-filtered* on MMLongBench only. Paper-filtering — required to
score a mixed corpus correctly, and the realistic single-document-QA
setting — shows routing is a *minor* lever there; the large earlier gap
was largely compensating for un-scoped retrieval. The shipped
reranker/classifier changes (ADR 0013/0014) do not harm, but the
*importance* of routing is scope-dependent and was overstated for the
document-scoped case.

**The open problem is the metric, not the set's balance.** A
discriminating eval needs sensitivity where it matters — chunk-level
recall (not page), or judged answer-correctness — plus an explicit
decision on whether the product retrieves document-scoped or
corpus-wide (very different routing value). Rebalancing robust-v1
further will not fix a saturated metric. A methodology decision is
required before any further routing tuning.

## Related

- ADR 0013: the overfit finding that motivated this; do not promote the
  v2 prompt / always-visual without a fair set — this is that set.
- ADR 0007: the visual-leg-loses-on-text result the fairness argument relies on.
- ADR 0008: the routing machinery this set exists to judge.
