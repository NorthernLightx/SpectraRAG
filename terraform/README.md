# Terraform — Azure Container Apps deploy

Provisions:
- Resource Group
- Log Analytics workspace (Container Apps' log sink)
- Azure Container Registry (Basic SKU)
- Container Apps Environment + a single Container App (`<prefix>-<env>-api`)
- Key Vault + three placeholder secrets (operator sets real values via az CLI)

## One-time bootstrap (do this manually before the first `terraform init`)

State lives in an Azure Storage account that Terraform itself does NOT
manage — chicken-and-egg. Provision it once:

```bash
az login

PREFIX=ragdemo
LOC=westeurope

az group create -n $PREFIX-tfstate-rg -l $LOC
az storage account create \
  -n ${PREFIX}tfstate \
  -g $PREFIX-tfstate-rg -l $LOC \
  --sku Standard_LRS \
  --encryption-services blob
az storage container create -n tfstate \
  --account-name ${PREFIX}tfstate
```

## First init + plan

```bash
cd terraform
terraform init \
  -backend-config="resource_group_name=$PREFIX-tfstate-rg" \
  -backend-config="storage_account_name=${PREFIX}tfstate" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=rag.tfstate"

terraform plan -var "prefix=$PREFIX" -var "env=prod"
```

## Setting secret values

After the first apply, populate Key Vault secrets:

```bash
KV=$(terraform output -raw key_vault_name)
az keyvault secret set --vault-name $KV --name openrouter-api-key --value "$OPENROUTER_API_KEY"
az keyvault secret set --vault-name $KV --name anthropic-api-key --value "$ANTHROPIC_API_KEY"
az keyvault secret set --vault-name $KV --name sentry-dsn --value "$SENTRY_DSN"
```

The Container App's system-assigned identity has `Get` on these secrets;
they're injected as env vars on the container at start.

## Cost (approximate, westeurope)

- Container App, scale-to-zero, ~10 prod requests/day: <$1/mo
- ACR Basic: $5/mo
- Log Analytics PerGB2018, low traffic: <$3/mo
- Key Vault: pennies

Budget: ~$10/mo idle, ~$30/mo with steady demo traffic.

## GitHub Actions secrets

Set in GitHub repo → Settings → Secrets and variables → Actions:

- `AZURE_CLIENT_ID` — Service Principal (federated identity / OIDC).
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`
- `TF_BACKEND_RG` — name of the bootstrap state-storage resource group.
- `TF_BACKEND_SA` — name of the bootstrap state-storage account.

Configure the federated credential on the Service Principal:

```bash
az ad sp create-for-rbac --name rag-gha-sp --role Contributor --scopes "/subscriptions/$AZ_SUB"
# Then set up OIDC trust:
az ad app federated-credential create \
  --id $APP_ID \
  --parameters '{"name":"main","issuer":"https://token.actions.githubusercontent.com","subject":"repo:OWNER/REPO:ref:refs/heads/main","audiences":["api://AzureADTokenExchange"]}'
```
