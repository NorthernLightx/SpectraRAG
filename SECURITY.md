# Security policy

## Reporting a vulnerability

If you believe you've found a security issue in this project, **please do
not open a public issue**. Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/NorthernLightx/multi-modal-rag/security)
   of this repository.
2. Click **Report a vulnerability**.
3. Describe the issue, including reproduction steps and impact assessment.

You'll get an acknowledgement within ~7 days. Realistic timeline for a fix
depends on complexity — this is a personal research project, not a 24/7
production service.

## Scope

This is a personal research codebase. The threat model is narrower than a
typical production service:

- **In scope**: secret leaks in code or commit history, command injection
  / SSRF / SQL injection in `src/`, dependency vulnerabilities flagged by
  `gitleaks` / Dependabot, RAG-specific issues like prompt injection
  through indexed corpora that materially compromise the host.
- **Out of scope**: Denial of service against your own local Docker
  Compose stack, ML adversarial inputs that produce wrong-but-not-harmful
  answers, third-party services this project integrates with (report
  those upstream).

## Defensive measures already in place

- Secrets never committed to the repo. `.env*` is gitignored;
  `.env.example` is the only template (with placeholder values).
- Pre-commit `gitleaks` hook + a `security` GitHub Actions workflow scan
  every PR for accidental secret commits.
- `pyproject.toml` author info uses GitHub's no-reply email pattern so
  contributor identities aren't scraped from package metadata.
- Production secrets in the Terraform stack are stored in Azure Key
  Vault; values are set out-of-band via `az keyvault secret set`, not
  committed to IaC.
