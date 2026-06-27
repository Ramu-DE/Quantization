#!/bin/bash
# =============================================================================
# GCP GPU VM Provisioning & Deployment Script
# Provisions an NVIDIA T4 GPU VM and deploys the quantization backend
# =============================================================================
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated: gcloud auth login
#   2. A GCP project with billing enabled
#   3. GPU quota approved (request at: IAM & Admin > Quotas > search "NVIDIA T4")
#
# Usage:
#   chmod +x gcp-gpu-setup.sh
#   ./gcp-gpu-setup.sh
#
# Cost estimate: ~$0.35/hr for n1-standard-4 + 1x T4 GPU
# =============================================================================

set -euo pipefail

# --- Configuration (edit these) ---
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-quantization-gpu}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-n1-standard-4}"  # 4 vCPU, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "============================================="
echo " GCP GPU VM Deployment for Quantization App"
echo "============================================="
echo ""
echo "Project:  $PROJECT_ID"
echo "Zone:     $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine:  $MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE"
echo ""

# --- Step 1: Enable required APIs ---
echo "[1/5] Enabling required APIs..."
gcloud services enable compute.googleapis.com --project="$PROJECT_ID" 2>/dev/null || true

# --- Step 2: Create the VM ---
echo "[2/5] Creating GPU VM instance..."
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --tags="http-server,quantization-api" \
    --quiet

echo "   VM created. Waiting for boot..."
sleep 30

# --- Step 3: Create firewall rule for API access ---
echo "[3/5] Creating firewall rule for port 8000..."
gcloud compute firewall-rules create allow-quantization-api \
    --project="$PROJECT_ID" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:8000,tcp:3000 \
    --target-tags=quantization-api \
    --source-ranges=0.0.0.0/0 \
    --quiet 2>/dev/null || echo "   (firewall rule already exists)"

# --- Step 4: Wait for GPU driver installation and deploy ---
echo "[4/5] Deploying application to VM..."

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
set -e

echo '--- Waiting for NVIDIA driver...'
for i in \$(seq 1 30); do
    if nvidia-smi &>/dev/null; then
        echo 'GPU driver ready!'
        nvidia-smi
        break
    fi
    echo \"Waiting for GPU driver... (\$i/30)\"
    sleep 10
done

echo ''
echo '--- Installing system dependencies...'
sudo apt-get update -qq
sudo apt-get install -y -qq git curl

echo ''
echo '--- Installing uv (Python package manager)...'
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\"\$HOME/.local/bin:\$PATH\"

echo ''
echo '--- Cloning/uploading project...'
mkdir -p ~/quantization-app/backend
"

# Upload the backend code
echo "   Uploading backend code..."
gcloud compute scp --recurse \
    "$(dirname "$(dirname "$(readlink -f "$0")")")/backend" \
    "$INSTANCE_NAME":~/quantization-app/ \
    --zone="$ZONE" --project="$PROJECT_ID" --quiet

# Install and run
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
set -e
export PATH=\"\$HOME/.local/bin:\$PATH\"

cd ~/quantization-app/backend

echo ''
echo '--- Installing Python dependencies with CUDA support...'
uv sync

echo ''
echo '--- Verifying GPU detection...'
uv run python -c \"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA not detected - running on CPU')
\"

echo ''
echo '--- Starting backend server...'
# Kill any existing server
pkill -f 'uvicorn main:app' 2>/dev/null || true
sleep 2

# Start with CORS allowing all origins for remote access
export CORS_ORIGINS='*'
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > ~/backend.log 2>&1 &

echo 'Waiting for server to start...'
sleep 5

# Test health endpoint
curl -s http://localhost:8000/health && echo ''
curl -s http://localhost:8000/device-info && echo ''

echo ''
echo '=== Backend deployed successfully! ==='
"

# --- Step 5: Print access information ---
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "============================================="
echo " DEPLOYMENT COMPLETE"
echo "============================================="
echo ""
echo " API URL:      http://$EXTERNAL_IP:8000"
echo " API Docs:     http://$EXTERNAL_IP:8000/docs"
echo " Device Info:  http://$EXTERNAL_IP:8000/device-info"
echo " Health:       http://$EXTERNAL_IP:8000/health"
echo ""
echo " To connect your local frontend:"
echo "   cd quantization-app"
echo "   echo 'NEXT_PUBLIC_API_URL=http://$EXTERNAL_IP:8000' > .env.local"
echo "   npm run dev"
echo ""
echo " To SSH into the VM:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo " To view server logs:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f ~/backend.log'"
echo ""
echo " To STOP the VM (stop billing):"
echo "   gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo " To DELETE the VM (remove completely):"
echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo " Estimated cost: ~\$0.35/hr while running"
echo "============================================="
