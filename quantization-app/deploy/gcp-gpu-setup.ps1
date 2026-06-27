# =============================================================================
# GCP GPU VM Provisioning & Deployment Script (PowerShell)
# Provisions an NVIDIA T4 GPU VM and deploys the quantization backend
# =============================================================================
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated: gcloud auth login
#   2. A GCP project with billing enabled
#   3. GPU quota approved (IAM & Admin > Quotas > search "NVIDIA T4")
#
# Usage:
#   .\gcp-gpu-setup.ps1
#   .\gcp-gpu-setup.ps1 -Zone "us-west1-b" -MachineType "n1-standard-8"
#
# Cost: ~$0.35/hr for n1-standard-4 + 1x T4, ~$0.70/hr for n1-standard-8 + 1x T4
# =============================================================================

param(
    [string]$ProjectId = (gcloud config get-value project 2>$null),
    [string]$Zone = "us-central1-a",
    [string]$InstanceName = "quantization-gpu",
    [string]$MachineType = "n1-standard-4",
    [string]$GpuType = "nvidia-tesla-t4",
    [int]$GpuCount = 1,
    [string]$BootDiskSize = "100GB"
)

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " GCP GPU VM Deployment for Quantization App" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project:  $ProjectId"
Write-Host "Zone:     $Zone"
Write-Host "Instance: $InstanceName"
Write-Host "Machine:  $MachineType + ${GpuCount}x $GpuType"
Write-Host ""

# Step 1: Enable APIs
Write-Host "[1/5] Enabling Compute API..." -ForegroundColor Yellow
gcloud services enable compute.googleapis.com --project="$ProjectId" 2>$null

# Step 2: Create VM
Write-Host "[2/5] Creating GPU VM instance..." -ForegroundColor Yellow
gcloud compute instances create $InstanceName `
    --project="$ProjectId" `
    --zone="$Zone" `
    --machine-type="$MachineType" `
    --accelerator="type=$GpuType,count=$GpuCount" `
    --image-family="pytorch-latest-gpu" `
    --image-project="deeplearning-platform-release" `
    --boot-disk-size="$BootDiskSize" `
    --boot-disk-type=pd-ssd `
    --maintenance-policy=TERMINATE `
    --metadata="install-nvidia-driver=True" `
    --scopes="https://www.googleapis.com/auth/cloud-platform" `
    --tags="http-server,quantization-api" `
    --quiet

Write-Host "   VM created. Waiting 30s for boot..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Step 3: Firewall
Write-Host "[3/5] Creating firewall rule for ports 8000, 3000..." -ForegroundColor Yellow
try {
    gcloud compute firewall-rules create allow-quantization-api `
        --project="$ProjectId" `
        --direction=INGRESS `
        --action=ALLOW `
        --rules=tcp:8000,tcp:3000 `
        --target-tags=quantization-api `
        --source-ranges=0.0.0.0/0 `
        --quiet
} catch {
    Write-Host "   (firewall rule already exists)" -ForegroundColor Gray
}

# Step 4: Upload code and deploy
Write-Host "[4/5] Uploading backend code..." -ForegroundColor Yellow
$BackendPath = Join-Path $PSScriptRoot "..\backend"

gcloud compute scp --recurse `
    "$BackendPath" `
    "${InstanceName}:~/quantization-app/" `
    --zone="$Zone" --project="$ProjectId" --quiet

Write-Host "   Installing dependencies and starting server..." -ForegroundColor Yellow
$SetupScript = @'
set -e
export PATH="$HOME/.local/bin:$PATH"

echo "--- Waiting for NVIDIA driver..."
for i in $(seq 1 30); do
    if nvidia-smi &>/dev/null; then
        echo "GPU driver ready!"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        break
    fi
    echo "Waiting for GPU driver... ($i/30)"
    sleep 10
done

echo ""
echo "--- Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd ~/quantization-app/backend

echo ""
echo "--- Installing Python dependencies..."
uv sync

echo ""
echo "--- Verifying GPU detection..."
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "--- Starting backend server..."
pkill -f 'uvicorn main:app' 2>/dev/null || true
sleep 2
export CORS_ORIGINS='*'
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > ~/backend.log 2>&1 &
sleep 5

echo ""
curl -s http://localhost:8000/health
echo ""
curl -s http://localhost:8000/device-info
echo ""
echo "=== Server running! ==="
'@

gcloud compute ssh $InstanceName --zone="$Zone" --project="$ProjectId" --command="$SetupScript"

# Step 5: Print results
$ExternalIp = gcloud compute instances describe $InstanceName `
    --zone="$Zone" --project="$ProjectId" `
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host " API URL:      " -NoNewline; Write-Host "http://${ExternalIp}:8000" -ForegroundColor Cyan
Write-Host " API Docs:     " -NoNewline; Write-Host "http://${ExternalIp}:8000/docs" -ForegroundColor Cyan
Write-Host " Device Info:  " -NoNewline; Write-Host "http://${ExternalIp}:8000/device-info" -ForegroundColor Cyan
Write-Host ""
Write-Host " Connect your local frontend:" -ForegroundColor Yellow
Write-Host "   cd quantization-app"
Write-Host "   Set-Content .env.local 'NEXT_PUBLIC_API_URL=http://${ExternalIp}:8000'"
Write-Host "   npm run dev"
Write-Host ""
Write-Host " Management:" -ForegroundColor Yellow
Write-Host "   SSH:    gcloud compute ssh $InstanceName --zone=$Zone"
Write-Host "   Logs:   gcloud compute ssh $InstanceName --zone=$Zone --command='tail -f ~/backend.log'"
Write-Host "   Stop:   gcloud compute instances stop $InstanceName --zone=$Zone"
Write-Host "   Delete: gcloud compute instances delete $InstanceName --zone=$Zone"
Write-Host ""
Write-Host " Cost: ~`$0.35/hr while running" -ForegroundColor Red
Write-Host "=============================================" -ForegroundColor Green
