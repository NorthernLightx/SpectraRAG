#Requires -Version 7
<#
.SYNOPSIS
    Bake the demo corpus into a Docker image and push it to GHCR.

.DESCRIPTION
    Renders page PNGs for the visual leg, builds a Qdrant snapshot via
    `bootstrap_corpus` (using qdrant-client's embedded `path:` mode), then
    runs `docker build` + `docker push` to GHCR with a `:baked-<short-sha>`
    tag. The bake artefacts are baked INTO the image so the deploy needs
    no external Qdrant or pages service.

    Re-running is safe — render_pages and bootstrap_corpus are idempotent.
    The printed tag is what `terraform apply` (or the deploy workflow)
    should pin via `-var "image_tag=..."`.

.PARAMETER PdfDir
    Directory of .pdf files to bake (default: data/papers).

.PARAMETER Collection
    Qdrant collection name; must match `RAG_CORPUS_COLLECTION` at runtime
    (default: rag_corpus).

.PARAMETER SkipPush
    Build the image locally but don't push. Useful for dry-runs.

.PREREQUISITES
    1. `docker compose up -d ollama` with bge-m3 pulled
       (`docker exec rag-ollama ollama pull bge-m3`).
    2. `docker login ghcr.io -u <gh-username> -p <github-pat>` already done.
       The PAT needs the `write:packages` scope.
    3. PDFs present in $PdfDir.
#>

[CmdletBinding()]
param(
    [string]$PdfDir = "data/papers",
    [string]$Collection = "rag_corpus",
    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed (exit $LASTEXITCODE)"
    }
}

# Derive GHCR repository from the git remote so forks Just Work.
$remoteUrl = git config --get remote.origin.url
if (-not $remoteUrl -or $remoteUrl -notmatch "github\.com[:/](?<owner>[^/]+)/(?<name>[^.]+?)(\.git)?$") {
    throw "Cannot derive GitHub owner/repo from git remote: '$remoteUrl'"
}
$registry = "ghcr.io/$($Matches.owner.ToLower())/$($Matches.name.ToLower())"

git diff --quiet HEAD
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Working tree has uncommitted changes — image tag won't match a public commit."
}
$shortSha = (git rev-parse --short HEAD).Trim()
$tag = "baked-$shortSha"
$image = "${registry}:${tag}"
Write-Host "Target: $image" -ForegroundColor Green

$pdfs = @(Get-ChildItem $PdfDir -Filter "*.pdf" -ErrorAction SilentlyContinue)
if ($pdfs.Count -eq 0) {
    throw "No .pdf files in $PdfDir"
}
Write-Host "Found $($pdfs.Count) PDFs in $PdfDir"

Invoke-Step "render_pages" {
    uv run python -m scripts.render_pages --pdf-dir $PdfDir --out-dir data/pages
}

Invoke-Step "bootstrap_corpus (path:./qdrant_local)" {
    uv run python -m scripts.bootstrap_corpus `
        --pdf-dir $PdfDir `
        --qdrant "path:./qdrant_local" `
        --collection $Collection
}

# Sanity-check the snapshot wrote real data — qdrant-client's local mode
# silently no-ops on some misconfigurations and we don't want to push an
# image with an empty index.
$qdrantBytes = (Get-ChildItem qdrant_local -Recurse -Force -File -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
if (($null -eq $qdrantBytes) -or ($qdrantBytes -lt 1MB)) {
    throw "qdrant_local/ is < 1 MB after bake — index didn't write."
}
Write-Host "Qdrant snapshot: $([math]::Round($qdrantBytes / 1MB, 1)) MB"

Invoke-Step "docker build" {
    docker build -t $image .
}

if ($SkipPush) {
    Write-Host "`n[SkipPush] Image built locally; not pushed." -ForegroundColor Yellow
    Write-Host "Tag: $tag"
    return
}

Invoke-Step "docker push" {
    docker push $image
}

Write-Host "`nPushed $image" -ForegroundColor Green
Write-Host "Deploy with:" -ForegroundColor Yellow
Write-Host "  terraform apply -var ""image_tag=$tag""" -ForegroundColor Yellow
Write-Host "or via the deploy.yml workflow with image_tag=$tag" -ForegroundColor Yellow
