# STATUS.md

Rolling 1-page summary of *current* project state. Append-only at phase boundaries — don't reflexively update after every run; let `data/eval/runs/` and ADRs hold per-run detail.

**Last updated:** 2026-05-02.

## Current production stack

```
PDF → PyMuPDF → section-aware chunking (1200 char, 200 overlap)
   → Ollama bge-m3 embeddings  (NaN-safe)
   → Qdrant (dense)  +  in-memory BM25  →  RRF fusion
   → BGE-v2-m3 cross-encoder rerank on GPU (top-50 → top-10)
   → qwen2.5:7b generator (Ollama, answer.yaml v4)
   → qwen2.5:7b LLM judge (faith / ans_rel / ctx_prec)
```

Run via `python -m scripts.eval_run --pdf <pdfs> --golden <yaml> --rerank --generate --judge`. Visual path is a sibling — `python -m scripts.eval_visual`, not part of the default flow.

## Current baseline

`data/eval/baseline.json` — run id `7b5242df5b38` (`run-20260501-190915.json`).

Golden v2: 23 queries × 5 ArXiv papers (17 in-corpus, 6 OOC). New-paper queries (q16–q23) are draft and need user review.

| Metric | Value |
|---|---|
| nDCG@5 (in-corpus macro) | 0.7214 |
| recall@10 (in-corpus macro) | 0.9412 |
| MRR (in-corpus macro) | 0.7437 |
| citation grounding | 1.0000 |
| faithfulness (LLM judge) | 0.8587 |
| answer relevance (LLM judge) | 0.8261 |
| context precision (LLM judge) | 0.6304 |
| p50 whole-query latency | ~73 s |
| p50 rerank stage on GPU | ~5.5 s |

CI regression gate fails the build if any of these drops by > 5%.

## Phase status

| Phase | State | ADR / pointer |
|---|---|---|
| 1 — text-only baseline | ✅ closed 2026-05-01 | — (baseline.json) |
| 2.0 — figure + table extraction | ✅ accepted, opt-in default-off | `0002-phase2-multimodal-chunks.md` |
| 2.1 — VLM captioning | ✅ accepted, opt-in default-off (recommended VLM `minicpm-v:8b`) | `0002-phase2-multimodal-chunks.md` |
| 2.2 — query expansion (rewrite/HyDE/combo) | ❌ rejected default-off, kept in tree | `0003-phase22-query-expansion.md` |
| 3 — visual retrieval (ColQwen2) | ✅ accepted as complementary path, separate CLI | `0004-phase3-visual-retrieval.md` |
| 4 — production polish (Azure, OTel, Sentry) | 🟡 scaffold landed (deploy + OTel + Sentry); apply not yet run | `0005-phase4-deploy-and-observability.md` |

Earlier rejected: contextual retrieval (Anthropic-style blurbs) — `0001-contextual-retrieval.md`. Rejected because it adds zero marginal value when reranker is in the pipeline.

## What's optional (opt-in via CLI flags)

| Flag | What it does | Status |
|---|---|---|
| `--rerank` (+ `--rerank-input-size`, `--rerank-device`) | BGE cross-encoder rerank | **default ON in production stack** |
| `--generate` + `--judge` | LLM generation + RAGAS-style judging | default ON for full eval |
| `--extract-figures --extract-tables` | Phase 2.0 multi-modal chunks | opt-in (ADR 0002) |
| `--vlm-caption-model minicpm-v:8b` | Phase 2.1 VLM captions | opt-in (ADR 0002) |
| `--query-expansion --query-expansion-mode rewrite` | Phase 2.2 LLM query rewrites | opt-in (ADR 0003 rejected by default) |
| `--contextualize --contextualize-provider {ollama,openrouter}` | Anthropic-style contextual retrieval | opt-in (ADR 0001 rejected) |
| `--postgres-dsn …` | Persist run to Postgres | opt-in |
| `scripts/eval_visual.py` | Phase 3 ColQwen2 path (separate CLI) | opt-in (ADR 0004) |

## Open questions / next steps

- **Cloud judge calibration.** `qwen2.5:7b` as judge is inconsistent on refusals (q15/q19/q21 correctly refused but faithfulness scored 0.0; q23 refusal scored ans_rel=0.0). A cloud judge (gpt-4o-mini at ~$0.001/query) would resolve the cp ambiguity in ADRs 0002 and the OOC scoring inconsistencies.
- **Hybrid text + visual.** ADR 0004 calls this out — text for definitional precision, visual for multi-hop / term-mismatch coverage. RRF-fuse top-K from both. Natural Phase 3.1 if/when chasing the combined number.
- **Per-query-category routing for query expansion.** ADR 0003's q4/q11 wins are real; rewriting only multi-hop / term-mismatch queries (gated by a small classifier) would surface them without the q9/q12 cost.
- **OOC refusal hardening.** q5/q23 don't refuse cleanly under `answer.yaml` v4. Durable fix is a rerank-score threshold gate (`if all top-K rerank scores < τ → return refusal directly`).
- **Golden v2 user review.** Queries q16–q23 (4 new papers) are draft; chunk-id verification was partial.
- **Phase 4 follow-ups.** Scaffold landed (ADR 0005); pending: first `terraform apply` against a real Azure subscription, soak run, then a follow-up PR to flip `deploy.yml` to auto-apply on push-to-main. PII redaction processor and full `timed_event` → span migration tracked separately.

## Hardware notes

- **GPU:** RTX 3070 8 GB. PyTorch is `torch 2.6.0+cu124` (downgraded from 2.11+cu126 by colpali-engine's transformers/peft constraints — see ADR 0004).
- **Ollama** runs in Docker with NVIDIA passthrough (`docker exec rag-ollama nvidia-smi`). Verify `curl localhost:11434/api/ps` shows `size_vram > 0`.
- **VRAM budget:** bge-m3 ~1.2 GB + reranker ~600 MB + qwen2.5:7b ~5 GB = ~7 GB. ColQwen2 alone is ~5 GB and can't co-exist with the text stack — visual path is run separately.
- **Local Postgres on :5432** is blocked by Hyper-V port exclusion on this dev box; storage end-to-end verified against file-backed SQLite. Production deploys won't hit this.
