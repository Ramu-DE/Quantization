# AI Model Quantization — Real GPU Testing

Interactive quantization visualizer with real LLM quantization on NVIDIA L4 GPU via Google Cloud Run.

## What This Does

Quantizes a real 1.1 billion parameter language model (TinyLlama) on GPU and visualizes exactly what happens to the weights, predictions, and text quality.

### Key Results (from actual GPU testing)

| Method | Size | Compression | Quality Loss | Speed |
|--------|------|-------------|--------------|-------|
| FP32 (original) | 4.1 GB | 1x | — | — |
| INT8 Symmetric | 1.0 GB | 4x | +1.5% perplexity | 0.92s |
| INT4 Symmetric | 525 MB | 8x | +2.9% perplexity | 1.07s |
| INT3 Symmetric | 393 MB | 10.7x | +421% (broken) | 0.93s |

**Generation speed:** 37 tokens/sec on NVIDIA L4 (90x faster than CPU)

## Architecture

```
quantization-app/
├── backend/                    # FastAPI + PyTorch (CUDA-enabled)
│   ├── main.py                # API with /device-info endpoint
│   ├── services/
│   │   ├── device.py          # GPU/CPU auto-detection
│   │   ├── real_quantization.py  # TinyLlama quantization engine
│   │   ├── gptq_real.py       # GPTQ with Hessian on GPU
│   │   └── calibration.py     # Wikitext-2 calibration loader
│   └── routers/
│       └── real_model.py      # All Real LLM API endpoints
├── src/                        # Next.js frontend (React + Tailwind + Recharts)
│   └── components/visualizers/
│       └── RealModelQuantization.tsx  # GPU visualization panel
├── deploy/                     # GCP deployment scripts
│   ├── gcp-gpu-setup.ps1     # Windows PowerShell
│   ├── gcp-gpu-setup.sh      # Linux/Mac
│   └── gcp-manage.sh         # Start/stop/upload
└── docs/                       # Educational documentation
```

## Quick Start

### Backend (FastAPI)

```bash
cd quantization-app/backend
pip install uv && uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend (Next.js)

```bash
cd quantization-app
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

Open http://localhost:3000 → click "Real LLM" tab.

## GPU Deployment (Google Cloud Run)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build container
gcloud builds submit backend/ \
  --tag=REGION-docker.pkg.dev/PROJECT/REPO/backend:latest

# Deploy with L4 GPU
gcloud beta run deploy quantization-gpu \
  --image=REGION-docker.pkg.dev/PROJECT/REPO/backend:latest \
  --gpu=1 --gpu-type=nvidia-l4 --cpu=4 --memory=16Gi \
  --max-instances=1 --no-gpu-zonal-redundancy \
  --allow-unauthenticated --port=8080
```

Cost: ~$0 when idle (scales to zero), charges only per request.

## Visualizations

The app provides interactive visualizations:

1. **Number Line Snap** — real weight values snapping to quantization grid
2. **Weight Matrix Heatmap** — before/after like image compression
3. **Token Probability Shift** — how next-word predictions change
4. **Quantization Grid Resolution** — INT8 (256 levels) vs INT4 (16) vs INT3 (8)
5. **Error Distribution** — the noise profile from rounding
6. **Perplexity Comparison** — quality metric across all methods
7. **Layer Sensitivity** — which layers degrade most

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /device-info` | GPU detection status |
| `GET /api/real-model/info` | Load model, return stats |
| `POST /api/real-model/quantize` | Quantize + measure perplexity |
| `POST /api/real-model/compare` | Compare INT8/INT4/INT3 |
| `POST /api/real-model/deep-visualize` | Number line + heatmap + token probs |
| `POST /api/real-model/gptq` | GPTQ with calibration data |
| `POST /api/real-model/benchmark-speed` | Tokens/second measurement |
| `POST /api/real-model/sensitivity` | Per-layer sensitivity analysis |

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch 2.6, CUDA 12.6, Transformers
- **Frontend:** Next.js 15, React, Tailwind CSS, Recharts, shadcn/ui
- **GPU:** NVIDIA L4 (22.5 GB VRAM) via Google Cloud Run
- **Model:** TinyLlama-1.1B-Chat (HuggingFace)
- **Deployment:** Docker, Google Cloud Build, Artifact Registry

## Full Results

See [GPU_QUANTIZATION_TESTING.md](GPU_QUANTIZATION_TESTING.md) for detailed test results, tables, and findings.
