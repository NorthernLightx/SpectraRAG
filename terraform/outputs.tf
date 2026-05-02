output "container_app_fqdn" {
  description = "Public URL of the deployed API."
  value       = azurerm_container_app.api.ingress[0].fqdn
}

output "acr_login_server" {
  description = "Container registry login server (used by GHA push)."
  value       = azurerm_container_registry.rag.login_server
}

output "key_vault_name" {
  description = "Key Vault name (operator uses this for `az keyvault secret set`)."
  value       = azurerm_key_vault.rag.name
}

output "resource_group" {
  description = "Resource group name."
  value       = azurerm_resource_group.rag.name
}
