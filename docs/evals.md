# Evaluations

How to run the eval, read its output, and not regress in CI.

## Golden sets

Versioned YAML in [`data/golden/`](../data/golden/). One file per version.

| Version | Queries | Papers | Note |
|---|---|---|---|
| v1 | 5 | 1 | smoke set; `baseline.json` references it |
| v2 | 23 | 5 | first multi-paper expansion; production baseline |
| v3 | 39 | 20 | 20-paper expansion adding 16 figure/table queries for routing analysis |

Schema is `GoldenQuery` in [`src/types/eval.py`](../src/types/eval.py).
Categories `factual`, `multi_hop`, `figure`, `table`, `equation`,
`out_of_corpus`. Those are the same labels the per-query router emits, so
per-subset analysis cross-references cleanly.

## Metrics

Retrieval (macro over in-corpus queries; OOC excluded, since they're 0 by
construction):

- `nDCG@5`, `recall@10`, `MRR`. See `src/eval/metrics_retrieval.py`.

Generation (LLM-as-judge, prompts in `src/prompts/library/`):

- `faithfulness`: claims supported by context.
- `answer_relevance`: does the answer address the question.
- `context_precision`: fraction of retrieved chunks that are relevant.
- `answer_correctness`: fraction of the query's `expected_facts` covered
  by the answer (ADR 0019). Chunk-id-robust: judges the answer text, not
  retrieved ids, so it survives a chunker change like ADR 0017 without
  re-anchoring. Populated only when the `GoldenQuery` carries
  `expected_facts` and the answer is not a refusal.
- `citation_grounding`: programmatic, not LLM-judged.

OOC handling: a "Not stated in the provided context." answer to an
unanswerable question scores 1.0 on faithfulness + answer_relevance
(correctly refusing). Same answer to an answerable question scores 0.
An in-corpus refusal on a query with `expected_facts` scores 0 on
`answer_correctness` (refusal covers no facts); OOC and no-facts queries
leave `answer_correctness` None (metric not applicable).

## Running it

Smoke (~3¢ on `gpt-4o-mini`, ~1 min):

```bash
.venv/Scripts/python.exe -m scripts.eval_run \
    --pdf data/papers/<one>.pdf --golden data/golden/v1.yaml \
    --generate --generator-provider openrouter --generator-model openai/gpt-4o-mini \
    --judge --judge-provider openrouter --judge-model openai/gpt-4o-mini \
    --rerank
```

Full v3 with the per-query router (retrieval-only, ~45 min on a single GPU):

```bash
.venv/Scripts/python.exe -m scripts.eval_run \
    --pdf (Get-ChildItem data/papers/*.pdf | ForEach-Object { $_.FullName }) \
    --golden data/golden/v3.yaml --rerank --router \
    --postgres-dsn "" --output-dir data/eval/runs --collection eval_phase32_router
```

Outputs land in `data/eval/runs/run-<timestamp>.{json,md}` (gitignored).
The JSON is `EvalRun` from `src/types/eval.py`. `run_id` is a content
hash so identical config + per-query data produce the same id.

### Measuring what the service runs

The API and `eval_run` build their retriever through the same
`RetrievalConfig` (`src/rag/retrieval_config.py`). Every run JSON records it
under `config.retrieval_config` with a 12-character
`config.retrieval_fingerprint`, and `/health` reports the fingerprint of the
stack the server actually wired. Equal fingerprints mean the two retrieved
the same way; `check_regression` prints the knobs that differ when they don't.

To measure a served stack, name its profile instead of passing retrieval
flags. `cpu` is what the Cloud Run image and `spectrarag serve` run (in-process
bge-m3, the MiniLM reranker, a 20-candidate pool per leg):

```bash
.venv/Scripts/python.exe -m scripts.eval_run --profile cpu --router \
    --pdf ... --golden ... --skip-ingest --collection <corpus> \
    --visual-collection <corpus>_visual
```

`--profile` rejects the individual retrieval flags (`--rerank`,
`--rerank-model`, `--cascade` and the rest) rather than silently ignoring them.

### Every arm from one run

A run through the router records each leg's ranked chunk ids in
`per_query[].leg_chunk_ids`. `scripts/derive_arms.py` turns one such run into
text-only, visual-only and hybrid arms at any fusion weight, with no model
calls. The arms share their leg outputs, so a per-query difference between two
of them is the fusion policy and nothing else. Record the legs deeper than the
reported k (`--fusion-depth 50`) or the fusion only ever sees top-k pages from
each leg:

```bash
.venv/Scripts/python.exe -m scripts.eval_run --router --force-route hybrid \
    --fusion-depth 50 ... --output-dir data/eval/runs
.venv/Scripts/python.exe -m scripts.derive_arms --run data/eval/runs/run-<ts>.json \
    --golden data/golden/mmdocir-v1.yaml --out-dir data/eval/runs/arms \
    --weight 1 --weight 5
```

The derived runs are ordinary run JSONs, so `check_regression` and
`scripts/experiments/paired_arm_compare.py` read them as they are.

## Regression gate

`scripts/check_regression.py` compares two run JSONs and fails when any gated
metric drops more than the threshold (default 5%): exit 0 if all are within
threshold, 1 if any regressed, 2 on bad input.

### What CI runs, every commit

`scripts/eval_retrieval_ci.py` builds the stack the Cloud Run image serves,
from the same `cpu` profile (bge-m3 on CPU, BM25, RRF, the MiniLM reranker),
over the committed `qdrant_local/rag_corpus` snapshot. The visual leg cannot
run on the runner (ColQwen2, and a page index outside git), so it replays from
`data/eval/fixtures/visual-legs-v3.json` and the router fuses it live. One pass
scores **page-level** nDCG@5 / recall@10 / MRR for two arms: the hybrid arm
against `data/eval/baseline_retrieval_hybrid.json`, and the text arm, read off
the same run's text leg, against `data/eval/baseline_retrieval.json`:

```bash
uv run python -m scripts.eval_retrieval_ci --output data/eval/runs/retrieval-ci.json \
    --hybrid-output data/eval/runs/retrieval-ci-hybrid.json
uv run python -m scripts.check_regression \
    --baseline data/eval/baseline_retrieval.json \
    --candidate data/eval/runs/retrieval-ci.json \
    --metrics ndcg_at_5 recall_at_10 mrr --threshold 0.05 --per-query recall_at_10
uv run python -m scripts.check_regression \
    --baseline data/eval/baseline_retrieval_hybrid.json \
    --candidate data/eval/runs/retrieval-ci-hybrid.json \
    --metrics ndcg_at_5 recall_at_10 mrr --threshold 0.05 --per-query recall_at_10
```

No Ollama, no GPU, no LLM, and the same retrieved ids on every run. Because
the run is deterministic, the gate also fails on any single query losing
recall@10 (`--per-query`): on 31 in-corpus queries one query going from found
to missed moves the mean by about 0.03, which the 5% bar lets through.

Re-record the visual fixture with `scripts/record_visual_legs.py` when the page
index, the visual model or the golden set changes. The recording runs ColQwen2
in bf16 on CPU; the served encoder is fp32, so the replayed leg is the served
one to within bf16 rounding. A query missing from the fixture fails the run
rather than falling back to text. Metrics are page-level because `rag_corpus` is the shipped demo
corpus, periodically re-baked by the docling chunker (ADR 0017 / 0021), which
renumbers the `::cN` suffix. The v3 golden's chunk-level labels drift out of
sync with that bake, but the page each one points at does not. Projecting both
sides to `paper::pN` coarsens the existing human labels rather than inventing
new ones, the same re-chunk-robustness reasoning behind ADR 0019's
`answer_correctness`. The visual (GPU) and generation/judge (API,
non-deterministic) legs are excluded here; they live in the full eval below.

The baselines are generated on a CPU dev box. If the runner ever fails the
per-query check from platform float differences alone (a gold page swapping
places with a near-tie at rank 10), regenerate both baselines on the runner.

Both baselines were regenerated on 2026-09-23 when the gate moved from an
unreranked text leg to the served stack; text-arm recall@10 went from 0.790 to
0.871 and no query lost a gold page.

### Full-stack baseline, manual or scheduled

`data/eval/baseline.json` is the end-to-end reference (v3 + router + visual +
extract-figures + extract-tables + paper-id-filter + region-number-boost +
rerank-length-norm + VLM-caption (gemma3:4b) + generate + judge; baseline
`83da5d51e4c3`, run 2026-05-11, replaced `f844619927e0`; ADR 0009 + 2nd
follow-up). It needs a GPU and Ollama, so it runs by hand (or on a scheduled
GPU runner), not per commit:

```bash
.venv/Scripts/python.exe -m scripts.check_regression \
    --baseline data/eval/baseline.json \
    --candidate data/eval/runs/run-<latest>.json \
    --threshold 0.05
```

Gated metrics there: the six listed under [Metrics](#metrics) (excluding
`citation_grounding`).

## Rebaselining

When an improvement is intentional:

1. Run the eval that produced it.
2. Sanity-check the per-query Markdown. Is the gain real or a
   measurement artefact?
3. `cp data/eval/runs/run-<id>.json data/eval/baseline.json`.
4. Commit alongside the change that justified it. Reference both in the
   commit message + an ADR if non-obvious.

Don't rebaseline silently. If the gate fired, that's worth a paragraph
of context.

## Ingestion scorecard

The metrics above score *answers*. `scripts/eval_ingestion.py` scores the
*chunked corpus*: structural quality, no LLM, no RAG pipeline, runs in
seconds so it can steer ingestion changes early (the gap ADR 0018's review
surfaced):

```bash
uv run python -m scripts.eval_ingestion --tag main          # write snapshot
uv run python -m scripts.eval_ingestion --tag wip --diff main  # show the delta
```

Tracks chunk count, length distribution, fragmentation, section-attribution
coverage, distinct sections, cross-page %. Snapshots commit to
`data/eval/ingestion/<tag>.json`; the Markdown writes per-category example
chunks so a moved metric is explainable, not just a number. `data/eval/
ingestion/main.json` is the post-ADR-0017 reference. Graph (entities/
relations, isolates, community shape) and bib-filter precision dimensions
are added by the GraphRAG spike; bib-filter ground truth is human-labelled
(the machine never authors truth; see `promote_candidates.py`).

## Figure-caption invariant guard

A deterministic structural guard, not a labelled set, so it stays inside the
machine-never-authors-truth rule. It encodes one rule from ADR 0022: a picture
whose caption starts with a primary `Figure N` / `Fig. N` / `Table N` / `Tab. N`
label must never be *surfaced* as `role=unlabeled`. A captioned figure is never
hidden behind the gallery's "unknown" bucket. (The bug it pins: `2604.28177v1`
p13, a real captioned illustration shown as `unlabeled`.) "Surfaced" is the role
`figures._to_browse_item` returns; the stored 3-way role is left alone, since the
retrieval filter still needs `decoration` and `unlabeled`.

Two pytest arms, both in the fast pre-push subset (`pytest -m "not slow and not
integration"`), so a breach fails the push before it reaches CI:

- `tests/unit/test_figure_caption_invariant.py`: runs synthetic captioned
  chunks through the real `_to_browse_item` and the caption-first arm of
  `_classify_figure_role`. Microseconds; catches the *code* regressing.
- `tests/unit/test_figure_caption_invariant_corpus.py`: scans the committed
  `qdrant_local/rag_corpus` snapshot and asserts the same invariant over every
  baked figure/table chunk, so a bad `--force` re-bake (for example, reverting
  the `table → figure` map) turns red against the shipping data. Read-only
  scroll, ~0.5 s; skips if the snapshot is absent.
