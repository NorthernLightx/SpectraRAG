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

variable "image_tag" {
  description = "Docker image tag to deploy."
  type        = string
  default     = "latest"
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
