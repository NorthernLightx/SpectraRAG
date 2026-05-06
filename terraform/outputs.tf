output "container_app_fqdn" {
  description = "Public URL of the deployed API."
  value       = azurerm_container_app.api.ingress[0].fqdn
}

output "image_reference" {
  description = "Container image the app pulls from (GHCR — free, public)."
  value       = "${var.image_repository}:${var.image_tag}"
}

output "key_vault_name" {
  description = "Key Vault name (operator uses this for `az keyvault secret set`)."
  value       = azurerm_key_vault.rag.name
}

output "resource_group" {
  description = "Resource group name."
  value       = azurerm_resource_group.rag.name
}
