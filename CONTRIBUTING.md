# Contributing

Thanks for your interest. This is a portfolio / research project — drive-by
PRs are welcome, but please open an issue first for anything beyond a typo
or single-line fix so we can align on scope.

## Local setup

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker + Docker
Compose, and a CUDA-capable GPU if you want to run the visual or rerank
stack locally.

```bash
git clone https://github.com/NorthernLightx/multi-modal-rag.git
cd multi-modal-rag
uv sync --extra dev
cp .env.example .env
docker compose up -d qdrant postgres langfuse ollama
docker exec rag-ollama ollama pull bge-m3
```

## What CI runs (mirror locally before pushing)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests scripts
uv run pytest -v --cov=src --cov-report=term-missing
uv run python -m scripts.check_regression \
    --baseline data/eval/baseline.json \
    --candidate data/eval/baseline.json \
    --threshold 0.05
```

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/). The
pattern is `type(scope): subject`, lowercase, imperative mood. Common types:

- `feat(<scope>):` — new behavior
- `fix(<scope>):` — bug fix
- `docs(<scope>):` — README / ADR / docstring changes
- `test(<scope>):` — test-only changes
- `refactor(<scope>):` — no behavior change
- `chore:` — repo housekeeping
- `ci:` / `build:` — workflow / build config
- `perf(<scope>):` — perf-only changes

Examples from this repo:

```
feat(eval): --refusal-score-threshold CLI flag wires gate into eval_run
fix(ci): drop unused pull-requests:write permission from deploy workflow
docs(adr): 0006 — OOC refusal gate (accepted opt-in; judge artifact noted)
chore(terraform): apply fmt + commit provider lock; gitignore .terraform/
```

Keep commits **atomic** — each commit should pass `pytest -m "not integration"`
on its own. If a refactor and a feature ride together, split them.

## Architecture decisions

Any non-obvious choice gets an ADR in `docs/decisions/NNNN-<slug>.md`.
Mirror the structure of the existing ones (Status / Date / Phase / Context /
Implementation / Decision / Caveats / References). PRs that change behavior
in an area covered by an ADR should reference or supersede it.

## What NEVER goes in a commit

This repo is open-source by design and is occasionally shared in job
applications. Be defensive about leakage:

- ❌ **Secrets** — API keys, tokens, DSNs, private keys, OAuth secrets.
  `.env` is gitignored; `.env.example` is the only sanctioned secrets-ish
  file (with placeholder values only). Pre-commit `gitleaks` hook is the
  safety net, not a substitute for care.
- ❌ **Personal info** — your home directory paths (`/c/Users/<you>/...`,
  `/home/<you>/...`, `C:\Users\<you>\...`), your real-name email if you
  prefer pseudonymity, internal hostnames, customer data, anything you
  wouldn't paste into a public Gist. Use `~` or env vars in pasted shell
  commands.
- ❌ **Large binary blobs** — `data/papers/`, `data/pages/`, model weights,
  `.parquet`, anything > 1 MB unless it's a versioned artifact like
  `data/eval/baseline.json`. Use the fetch scripts to reproduce locally.
- ❌ **Ephemeral local state** — `.venv/`, `logs/`, `.coverage*`,
  `__pycache__/`, `qdrant_storage/`, `postgres_data/`, IDE configs,
  agent-tooling state (`.claude/`, `.cursor/`, `CLAUDE.md`,
  `docs/superpowers/`). The `.gitignore` covers these defensively.
- ❌ **Internal-only docs** — running-state notes (`STATUS.md`),
  scratchpad plans, agent-generated planning files. Keep them local; what
  belongs in the repo is the ADR after the decision is made.

If you accidentally commit something sensitive, **do not just delete it in a
follow-up commit** — the data stays in history. Use `git filter-repo` to
purge the file from every commit, then force-push and rotate the secret.
See [GitHub's removing sensitive data guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Pre-commit hooks

Recommended (mirrors CI + adds secret scanning):

```bash
uv pip install pre-commit
pre-commit install
```

After install, `git commit` runs ruff, gitleaks, basic file-shape checks
locally. CI runs the same set on push.

## Reporting security issues

See [SECURITY.md](./SECURITY.md). Please do not file public issues for
security-impacting bugs — use GitHub's private vulnerability reporting.
