#!/bin/bash
# Deploy Quantization Visualizer to Google Cloud Run with GPU
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Google Cloud project with billing enabled
#   3. Cloud Run API enabled: gcloud services enable run.googleapis.com
#   4. Artifact Registry API: gcloud services enable artifactregistry.googleapis.com
#
# Usage: ./cloud-run-deploy.sh <PROJECT_ID> [REGION]
#
# Example: ./cloud-run-deploy.sh my-project-123 us-central1

set -e

PROJECT_ID=${1:?"Usage: $0 <PROJECT_ID> [REGION]"}
REGION=${2:-"us-central1"}
SERVICE_NAME="quantization-visualizer"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "========================================="
echo "  Deploying Quantization Visualizer"
echo "  Project: ${PROJECT_ID}"
echo "  Region:  ${REGION}"
echo "  Image:   ${IMAGE_NAME}"
echo "========================================="

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo ""
echo "[1/5] Enabling APIs..."
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

# Build container image using Cloud Build (no local Docker needed)
echo ""
echo "[2/5] Building container image (this takes 10-15 minutes)..."
gcloud builds submit --tag ${IMAGE_NAME} --timeout=1800 ..

# Deploy to Cloud Run with GPU
echo ""
echo "[3/5] Deploying to Cloud Run with L4 GPU..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --region ${REGION} \
    --platform managed \
    --memory 16Gi \
    --cpu 4 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --timeout 300 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 1 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars "PYTHONUNBUFFERED=1"

# Get URL
echo ""
echo "[4/5] Getting service URL..."
URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')

echo ""
echo "[5/5] Testing deployment..."
curl -s "${URL}/health" && echo ""

echo ""
echo "========================================="
echo "  DEPLOYED SUCCESSFULLY!"
echo ""
echo "  Public URL: ${URL}"
echo "  API Docs:   ${URL}/docs"
echo ""
echo "  To update:  gcloud builds submit --tag ${IMAGE_NAME} .. && gcloud run deploy ${SERVICE_NAME} --image ${IMAGE_NAME} --region ${REGION}"
echo "  To delete:  gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
echo "========================================="
