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

# `:main` is the always-fresh tag pushed by the docker.yml `publish` job on
# every main-branch commit — it includes the rendered pages + Qdrant
# snapshot, baked in CI from the `data/curated_demo/papers.txt` manifest.
# Pin a specific commit for prod with `-var "image_tag=sha-abc1234"`.
variable "image_tag" {
  description = "Docker image tag to deploy (default :main; pin :sha-<short> to lock a specific build)."
  type        = string
  default     = "main"
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
