#!/usr/bin/env bash
# =============================================================================
# CHBIndex: one-time Google Cloud setup for the Cloud Run deploy
# =============================================================================
# Creates what .github/workflows/deploy.yml needs. Safe to rerun: every
# "create" skips what already exists, and IAM bindings are idempotent.
#
#   1. APIs: Cloud Run, Artifact Registry, IAM credentials, STS, BigQuery
#   2. Artifact Registry repo "chbindex" (keeps the 10 newest images per name)
#   3. Three service accounts, each with only what it needs:
#        chbindex-api        reads the BigQuery dataset, runs query jobs
#        chbindex-dashboard  no roles here; the deploy grants it run.invoker
#                            on the API service only
#        chbindex-deployer   deploys Cloud Run, pushes images, acts as the
#                            two runtime accounts
#   4. Workload Identity Federation: GitHub OIDC tokens from main of
#      $GITHUB_REPO (and nothing else) may act as chbindex-deployer
#
# Run from the repo root, logged in as a project owner:
#   gcloud auth login
#   bash scripts/gcp/setup_cloud_run.sh
# Override defaults with env vars: PROJECT_ID, REGION, GITHUB_REPO.
# =============================================================================

set -euo pipefail

# Never wait for a y/N answer: the existence checks below send their output
# to /dev/null, so a prompt there (e.g. "API not enabled yet, enable and
# retry?" while a just-enabled API propagates) would hang with nothing shown.
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT_ID="${PROJECT_ID:-elevation-465520}"
REGION="${REGION:-us-central1}"
GITHUB_REPO="${GITHUB_REPO:-sanket1009/chbindex}"
DATASET="${DATASET:-chbindex_06037}"
POOL="github"
PROVIDER="github-main"

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')
API_SA="chbindex-api@${PROJECT_ID}.iam.gserviceaccount.com"
DASH_SA="chbindex-dashboard@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_SA="chbindex-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "== 1. APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com bigquery.googleapis.com

echo "== 2. Artifact Registry"
if ! gcloud artifacts repositories describe chbindex --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create chbindex --repository-format docker \
    --location "$REGION" --description "CHBIndex API and dashboard images"
fi
# Storage costs money: keep only the 10 newest images of each name
POLICY=$(mktemp)
cat > "$POLICY" <<'JSON'
[
  {"name": "keep-10-newest", "action": {"type": "Keep"},
   "mostRecentVersions": {"keepCount": 10}},
  {"name": "delete-older", "action": {"type": "Delete"},
   "condition": {"tagState": "ANY", "olderThan": "1d"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies chbindex --location "$REGION" \
  --policy "$POLICY" --no-dry-run
rm -f "$POLICY"

echo "== 3. Service accounts"
for sa in chbindex-api chbindex-dashboard chbindex-deployer; do
  if ! gcloud iam service-accounts describe "${sa}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$sa" --display-name "$sa"
  fi
done

# API: run query jobs in the project, read one dataset. readSessionUser lets
# to_dataframe() use the BigQuery Storage Read API (google-cloud-bigquery-storage)
for role in roles/bigquery.jobUser roles/bigquery.readSessionUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition None \
    --member "serviceAccount:$API_SA" --role "$role" >/dev/null
done
# Dataset-level IAM through `bq add-iam-policy-binding` needs allowlisting, so
# read access is granted the classic way: an entry in the dataset's access
# list (role READER), with the BigQuery client from the project's .venv.
echo "Granting read access on ${PROJECT_ID}:${DATASET} to ${API_SA}"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
"$PY" - "$PROJECT_ID" "$DATASET" "$API_SA" <<'PYCODE'
import sys
from google.cloud import bigquery

project, dataset_id, member = sys.argv[1:]
client = bigquery.Client(project=project)
dataset = client.get_dataset(f"{project}.{dataset_id}")
entry = bigquery.AccessEntry(role="READER", entity_type="userByEmail", entity_id=member)
entries = list(dataset.access_entries)
if any(e.entity_id == member and e.role == "READER" for e in entries):
    print("  already has read access")
else:
    dataset.access_entries = [*entries, entry]
    client.update_dataset(dataset, ["access_entries"])
    print("  granted READER")
PYCODE

# Deployer: manage Cloud Run services, push to this one repo, act as the runtime accounts
gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition None \
  --member "serviceAccount:$DEPLOY_SA" --role roles/run.admin >/dev/null
gcloud artifacts repositories add-iam-policy-binding chbindex --location "$REGION" \
  --member "serviceAccount:$DEPLOY_SA" --role roles/artifactregistry.writer >/dev/null
for sa in "$API_SA" "$DASH_SA"; do
  gcloud iam service-accounts add-iam-policy-binding "$sa" \
    --member "serviceAccount:$DEPLOY_SA" --role roles/iam.serviceAccountUser >/dev/null
done

echo "== 4. Workload Identity Federation"
if ! gcloud iam workload-identity-pools describe "$POOL" --location global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --location global \
    --display-name "GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" \
    --workload-identity-pool "$POOL" --location global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --workload-identity-pool "$POOL" --location global \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${GITHUB_REPO}' && assertion.ref == 'refs/heads/main'"
fi
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  >/dev/null

cat <<EOF

Done. Set these GitHub repository variables (not secrets; none of them is a credential):
  GCP_PROJECT_ID   = ${PROJECT_ID}
  GCP_REGION       = ${REGION}
  GCP_WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_DEPLOYER_SA  = ${DEPLOY_SA}

With the GitHub CLI:
  gh variable set GCP_PROJECT_ID --body ${PROJECT_ID}
  gh variable set GCP_REGION --body ${REGION}
  gh variable set GCP_WIF_PROVIDER --body projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  gh variable set GCP_DEPLOYER_SA --body ${DEPLOY_SA}
EOF
