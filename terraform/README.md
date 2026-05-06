# Terraform — Azure Container Apps deploy

Provisions:
- Resource Group
- Log Analytics workspace (Container Apps' log sink)
- Container Apps Environment + a single Container App (`<prefix>-<env>-api`)
- Key Vault + three placeholder secrets (operator sets real values via az CLI)

The container image is pulled from GHCR (the public image
`.github/workflows/docker.yml` already pushes to). No Azure Container
Registry — saves the $5/mo Basic SKU for a portfolio-traffic deploy.

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

terraform plan -var "prefix=$PREFIX" -var "env=prod" -var "image_tag=baked-XXXXXXX"
```

`image_tag` has no default and `terraform plan` will fail without it — the
default `:main` tag would deploy a code-only image (the `qdrant_local/` and
`data/pages/` bake artefacts are gitignored, so CI can't include them).

## Baking the deploy image

The Container App pulls the public image from
`ghcr.io/<owner>/<repo>:<tag>`. CI (`.github/workflows/docker.yml`) only
publishes on version tags (`v*`); the actual demo image is published from
a developer machine that has the rendered page PNGs and Qdrant snapshot.

Prereqs (one-time):

```bash
docker compose up -d ollama
docker exec rag-ollama ollama pull bge-m3
echo $GITHUB_PAT | docker login ghcr.io -u <gh-user> --password-stdin
```

Bake + push:

```powershell
.\scripts\Publish-DemoImage.ps1
```

The script renders pages, builds the Qdrant snapshot via
`bootstrap_corpus --qdrant path:./qdrant_local`, runs `docker build` +
`docker push`, and prints the tag (e.g. `baked-693f6d4`). Pin that tag in
the next `terraform apply` or in the deploy workflow input.

## First apply

```bash
terraform apply \
  -var "prefix=$PREFIX" \
  -var "env=prod" \
  -var "image_tag=baked-693f6d4"
```

Re-baking is idempotent — re-running `Publish-DemoImage.ps1` only rebuilds the
layers whose source changed, and the qdrant snapshot is skipped when the
collection is already populated (override with `--force` from the script's
internals if the corpus content changed).

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

- Container App, scale-to-zero, portfolio traffic: $0 (within always-free
  quota: 180k vCPU-sec + 2M requests/mo)
- GHCR (image registry): $0 (free for public repos)
- Log Analytics: $0 at low traffic (5 GB/mo free tier)
- Key Vault: pennies, within free 10k-ops/mo tier

Budget: ~$0/mo at portfolio traffic. The $200 Azure free-trial credit
covers any incidental usage in the first 30 days; after that, staying
within the free tiers above keeps the bill at $0.

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
