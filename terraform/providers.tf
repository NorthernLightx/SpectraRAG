terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.10"
    }
  }
  backend "azurerm" {
    # Backend values supplied at `terraform init -backend-config=...`.
    # See terraform/README.md for the bootstrap script.
  }
}

provider "azurerm" {
  features {}
}
