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
`out_of_corpus` — same labels the per-query router emits, so per-subset
analysis cross-references cleanly.

## Metrics

Retrieval (macro over in-corpus queries; OOC excluded — they're 0 by
construction):

- `nDCG@5`, `recall@10`, `MRR` — see `src/eval/metrics_retrieval.py`.

Generation (LLM-as-judge, prompts in `src/prompts/library/`):

- `faithfulness` — claims supported by context.
- `answer_relevance` — does the answer address the question.
- `context_precision` — fraction of retrieved chunks that are relevant.
- `citation_grounding` — programmatic, not LLM-judged.

OOC handling: a "Not stated in the provided context." answer to an
unanswerable question scores 1.0 on faithfulness + answer_relevance
(correctly refusing). Same answer to an answerable question scores 0.

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
The JSON is `EvalRun` from `src/types/eval.py` — `run_id` is a content
hash so identical config + per-query data produce the same id.

## Regression gate

`scripts/check_regression.py` compares two run JSONs. CI runs it on every
push against `data/eval/baseline.json` (currently v2 baseline
`7b5242df5b38`):

```bash
.venv/Scripts/python.exe -m scripts.check_regression \
    --baseline data/eval/baseline.json \
    --candidate data/eval/runs/run-<latest>.json \
    --threshold 0.05
```

Exit 0 if all gated metrics within 5%, 1 if any regressed, 2 on bad input.
Gated metrics: the six listed above (excluding `citation_grounding`).

The gate is currently a smoke-test in CI (compares baseline against
itself) until self-hosted runners can execute the live stack — see
`.github/workflows/ci.yml` "Eval regression gate" step.

## Rebaselining

When an improvement is intentional:

1. Run the eval that produced it.
2. Sanity-check the per-query Markdown — is the gain real or a
   measurement artefact?
3. `cp data/eval/runs/run-<id>.json data/eval/baseline.json`.
4. Commit alongside the change that justified it. Reference both in the
   commit message + an ADR if non-obvious.

Don't rebaseline silently — if the gate fired, that's worth a paragraph
of context.
