#!/bin/bash
# GoldenMatch — Vertex AI Setup Script
#
# Sets up Google Cloud credentials for Vertex AI embeddings.
# Run this once per machine. No GPU required.
#
# Usage:
#   bash scripts/setup_vertex_ai.sh
#
# Prerequisites:
#   - gcloud CLI installed: https://cloud.google.com/sdk/docs/install
#   - A Google Cloud project with billing enabled

set -e

echo ""
echo "  ⚡ GoldenMatch — Vertex AI Setup"
echo "  ================================"
echo ""

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    echo "  ERROR: gcloud CLI not found."
    echo "  Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get or set project
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "  No GCP project set. Available projects:"
    gcloud projects list --format="value(projectId)"
    echo ""
    read -p "  Enter your project ID: " PROJECT
    gcloud config set project "$PROJECT"
fi
echo "  Project: $PROJECT"

# Enable Vertex AI API
echo ""
echo "  Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com --project="$PROJECT" 2>/dev/null || true
echo "  ✓ Vertex AI API enabled"

# Authenticate
echo ""
echo "  Setting up authentication..."
echo "  Option 1: User credentials (for local dev)"
echo "  Option 2: Service account (for production/CI)"
echo ""
read -p "  Choose [1/2] (default: 1): " AUTH_CHOICE

if [ "$AUTH_CHOICE" = "2" ]; then
    SA_NAME="goldenmatch-gpu"
    SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
    KEY_PATH="$(pwd)/.testing/google-service-account.json"

    echo "  Creating service account: $SA_EMAIL"
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="GoldenMatch GPU" \
        --project="$PROJECT" 2>/dev/null || echo "  (already exists)"

    echo "  Generating key..."
    mkdir -p .testing
    gcloud iam service-accounts keys create "$KEY_PATH" \
        --iam-account="$SA_EMAIL" \
        --project="$PROJECT"

    echo "  Granting permissions..."
    gcloud projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/aiplatform.user" \
        --project="$PROJECT" --quiet

    echo ""
    echo "  ✓ Service account key saved to: $KEY_PATH"
    echo ""
    echo "  Add to your shell profile:"
    echo "    export GOOGLE_APPLICATION_CREDENTIALS=$KEY_PATH"
    echo "    export GOOGLE_CLOUD_PROJECT=$PROJECT"
else
    gcloud auth application-default login
    echo ""
    echo "  ✓ User credentials configured"
    echo ""
    echo "  Add to your shell profile:"
    echo "    export GOOGLE_CLOUD_PROJECT=$PROJECT"
fi

# Test
echo ""
echo "  Testing Vertex AI connection..."
python -c "
from goldenmatch.core.vertex_embedder import VertexEmbedder
import os
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', '$PROJECT')
e = VertexEmbedder()
result = e.embed_column(['test embedding'], cache_key='setup_test')
print(f'  ✓ Vertex AI working! Embedding dim: {result.shape[1]}')
" 2>/dev/null || echo "  ⚠ Test failed — check credentials and project ID"

echo ""
echo "  Setup complete! Run GoldenMatch:"
echo "    export GOOGLE_CLOUD_PROJECT=$PROJECT"
echo "    goldenmatch dedupe your_file.csv"
echo ""
