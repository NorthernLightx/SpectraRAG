variable "prefix" {
  description = "Resource name prefix (3-12 chars, lowercase)."
  type        = string
  default     = "ragdemo"
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "westeurope"
}

variable "env" {
  description = "Deployment environment (prod | staging | dev)."
  type        = string
  default     = "prod"
}

# Image tags follow the `baked-<short-sha>` convention published by
# `scripts/bake_and_push.ps1` — the bake script prints the exact tag it
# pushed. There's intentionally no default: a stale `:main` or `:latest`
# would deploy a code-only image (no qdrant_local, no data/pages) that
# 503s every request. Force the operator to pick the tag they baked.
variable "image_tag" {
  description = "Docker image tag to deploy (e.g. baked-693f6d4 from scripts/bake_and_push.ps1)."
  type        = string
}

# Image lives on GHCR (free for public repos) instead of an Azure Container
# Registry — saves $5/mo on the Basic SKU and the image was already being
# pushed there by `.github/workflows/docker.yml`. The `image_repository`
# default points at the public repo this terraform was authored for; override
# for forks via `terraform plan -var image_repository=ghcr.io/<owner>/<name>`.
variable "image_repository" {
  description = "Public container image repository (GHCR or DockerHub)."
  type        = string
  default     = "ghcr.io/northernlightx/multi-modal-rag"
}

variable "min_replicas" {
  type    = number
  default = 0
}

variable "max_replicas" {
  type    = number
  default = 2
}

variable "cpu" {
  type    = number
  default = 0.5
}

variable "memory" {
  type    = string
  default = "1Gi"
}
