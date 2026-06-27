# Google Cloud Run Deployment Guide

## Quick Deploy (3 commands)

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build & Deploy (with GPU - recommended)
cd deploy
chmod +x cloud-run-deploy.sh
./cloud-run-deploy.sh YOUR_PROJECT_ID us-central1

# OR without GPU (cheaper):
./cloud-run-no-gpu.sh YOUR_PROJECT_ID us-central1
```

That's it. You'll get a public URL like `https://quantization-visualizer-xxxxx-uc.a.run.app`

---

## Options Comparison

| Option | Cost (idle) | Cost (active) | Real LLM Speed | Cold Start |
|--------|-------------|---------------|----------------|------------|
| **With L4 GPU** | $0 | ~$1.50/hour | 30-80 tok/s | ~60s |
| **CPU only (16GB)** | $0 | ~$0.30/hour | 5 tok/s | ~30s |

Both scale to zero when not used (no idle cost).

---

## Prerequisites

1. **Google Cloud account** with billing enabled
2. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
3. **Project** with these APIs enabled:
   - Cloud Run
   - Container Registry (or Artifact Registry)
   - Cloud Build

Enable all at once:
```bash
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com
```

---

## Step-by-Step Manual Deployment

### 1. Build the Container

```bash
# From the quantization-app/ directory
gcloud builds submit --tag gcr.io/YOUR_PROJECT/quantization-visualizer --timeout=1800
```

This takes ~10-15 minutes (downloads TinyLlama model into the image).

### 2. Deploy to Cloud Run

**With GPU (L4):**
```bash
gcloud run deploy quantization-visualizer \
    --image gcr.io/YOUR_PROJECT/quantization-visualizer \
    --region us-central1 \
    --memory 16Gi \
    --cpu 4 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 1 \
    --port 8080 \
    --allow-unauthenticated
```

**Without GPU:**
```bash
gcloud run deploy quantization-visualizer \
    --image gcr.io/YOUR_PROJECT/quantization-visualizer \
    --region us-central1 \
    --memory 16Gi \
    --cpu 4 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 1 \
    --port 8080 \
    --allow-unauthenticated
```

### 3. Get the URL

```bash
gcloud run services describe quantization-visualizer --region us-central1 --format='value(status.url)'
```

---

## Configuration

### Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Cloud Run injects this automatically |
| `NEXT_PUBLIC_API_URL` | `""` (same-origin) | Leave empty for Cloud Run |

### Scaling

```bash
# Keep warm (no cold start, costs ~$5/day)
gcloud run services update quantization-visualizer --min-instances 1

# Scale to zero (free when idle, 30-60s cold start)
gcloud run services update quantization-visualizer --min-instances 0
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "GPU quota exceeded" | Request L4 GPU quota: Cloud Console → IAM → Quotas → filter "NVIDIA L4" → Request increase |
| Build timeout | Increase: `--timeout=3600` |
| Cold start too slow | Set `--min-instances 1` (keeps one instance warm) |
| 502 errors | Check logs: `gcloud run services logs read quantization-visualizer` |
| OOM on Real LLM | Increase memory: `--memory 32Gi` |

---

## Cost Control

```bash
# Delete when done (stops all charges)
gcloud run services delete quantization-visualizer --region us-central1

# Or just scale to zero
gcloud run services update quantization-visualizer --min-instances 0 --max-instances 0
```

---

## GPU Availability by Region

L4 GPUs on Cloud Run are available in:
- `us-central1` (Iowa)
- `europe-west4` (Netherlands)  
- `asia-southeast1` (Singapore)

If L4 isn't available, use CPU-only deployment (still works, just slower).
