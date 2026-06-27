#!/bin/bash
# Deploy WITHOUT GPU (cheaper, CPU-only, still works but slower on Real LLM tab)
# Cost: ~$0 idle, ~$5-15/month with light use
#
# Usage: ./cloud-run-no-gpu.sh <PROJECT_ID> [REGION]

set -e

PROJECT_ID=${1:?"Usage: $0 <PROJECT_ID> [REGION]"}
REGION=${2:-"us-central1"}
SERVICE_NAME="quantization-visualizer"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Deploying (CPU-only, no GPU)..."
echo "Project: ${PROJECT_ID}, Region: ${REGION}"

gcloud config set project ${PROJECT_ID}
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

echo "Building image..."
gcloud builds submit --tag ${IMAGE_NAME} --timeout=1800 ..

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --region ${REGION} \
    --platform managed \
    --memory 16Gi \
    --cpu 4 \
    --timeout 300 \
    --concurrency 5 \
    --min-instances 0 \
    --max-instances 1 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars "PYTHONUNBUFFERED=1"

URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')
echo ""
echo "Deployed: ${URL}"
echo "API Docs: ${URL}/docs"
