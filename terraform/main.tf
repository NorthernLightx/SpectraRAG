locals {
  base_name = "${var.prefix}-${var.env}"
  tags = {
    project = "multi-modal-paper-rag"
    env     = var.env
    managed = "terraform"
  }
}

resource "azurerm_resource_group" "rag" {
  name     = "${local.base_name}-rg"
  location = var.location
  tags     = local.tags
}

resource "azurerm_log_analytics_workspace" "rag" {
  name                = "${local.base_name}-logs"
  location            = azurerm_resource_group.rag.location
  resource_group_name = azurerm_resource_group.rag.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_container_registry" "rag" {
  name                = replace("${local.base_name}acr", "-", "")
  resource_group_name = azurerm_resource_group.rag.name
  location            = azurerm_resource_group.rag.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

resource "azurerm_container_app_environment" "rag" {
  name                       = "${local.base_name}-cae"
  location                   = azurerm_resource_group.rag.location
  resource_group_name        = azurerm_resource_group.rag.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.rag.id
  tags                       = local.tags
}

resource "azurerm_container_app" "api" {
  name                         = "${local.base_name}-api"
  resource_group_name          = azurerm_resource_group.rag.name
  container_app_environment_id = azurerm_container_app_environment.rag.id
  revision_mode                = "Single"
  tags                         = local.tags

  registry {
    server   = azurerm_container_registry.rag.login_server
    username = azurerm_container_registry.rag.admin_username
    # gitleaks:allow — value is a Terraform secret NAME (string identifier),
    # not a hardcoded password. The actual credential value lives below in
    # the `secret` block where it references azurerm_container_registry.rag.admin_password.
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.rag.admin_password
  }

  secret {
    name                = "openrouter-api-key"
    key_vault_secret_id = azurerm_key_vault_secret.openrouter.id
    identity            = "System"
  }

  secret {
    name                = "anthropic-api-key"
    key_vault_secret_id = azurerm_key_vault_secret.anthropic.id
    identity            = "System"
  }

  secret {
    name                = "sentry-dsn"
    key_vault_secret_id = azurerm_key_vault_secret.sentry.id
    identity            = "System"
  }

  identity {
    type = "SystemAssigned"
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name   = "api"
      image  = "${azurerm_container_registry.rag.login_server}/multi-modal-paper-rag:${var.image_tag}"
      cpu    = var.cpu
      memory = var.memory

      env {
        name  = "RAG_ENV"
        value = var.env
      }
      env {
        name  = "RAG_LOG_LEVEL"
        value = "INFO"
      }
      env {
        name        = "OPENROUTER_API_KEY"
        secret_name = "openrouter-api-key"
      }
      env {
        name        = "ANTHROPIC_API_KEY"
        secret_name = "anthropic-api-key"
      }
      env {
        name        = "SENTRY_DSN"
        secret_name = "sentry-dsn"
      }
      env {
        name  = "SENTRY_ENVIRONMENT"
        value = var.env
      }
      env {
        name  = "SENTRY_TRACES_SAMPLE_RATE"
        value = "0.1"
      }
      env {
        name  = "OTEL_SERVICE_NAME"
        value = "rag-api-${var.env}"
      }
      # OTEL_EXPORTER_OTLP_ENDPOINT intentionally left unset by default —
      # operations sets it via az CLI (or future managed-grafana wiring).
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}
