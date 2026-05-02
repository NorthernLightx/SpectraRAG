data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "rag" {
  name                       = "${local.base_name}-kv"
  resource_group_name        = azurerm_resource_group.rag.name
  location                   = azurerm_resource_group.rag.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  purge_protection_enabled   = false
  soft_delete_retention_days = 7
  tags                       = local.tags

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge", "Recover",
    ]
  }
}

# Grant the Container App's system-assigned identity Get on secrets.
resource "azurerm_key_vault_access_policy" "container_app" {
  key_vault_id       = azurerm_key_vault.rag.id
  tenant_id          = data.azurerm_client_config.current.tenant_id
  object_id          = azurerm_container_app.api.identity[0].principal_id
  secret_permissions = ["Get"]
}

# Placeholder secret values: real values are set out-of-band by an operator
# via `az keyvault secret set`. Terraform tracks the *resource*, not the
# value — that's why the value is the literal string "set-out-of-band".
resource "azurerm_key_vault_secret" "openrouter" {
  name         = "openrouter-api-key"
  value        = "set-out-of-band"
  key_vault_id = azurerm_key_vault.rag.id
  lifecycle {
    ignore_changes = [value]
  }
}

resource "azurerm_key_vault_secret" "anthropic" {
  name         = "anthropic-api-key"
  value        = "set-out-of-band"
  key_vault_id = azurerm_key_vault.rag.id
  lifecycle {
    ignore_changes = [value]
  }
}

resource "azurerm_key_vault_secret" "sentry" {
  name         = "sentry-dsn"
  value        = "set-out-of-band"
  key_vault_id = azurerm_key_vault.rag.id
  lifecycle {
    ignore_changes = [value]
  }
}
