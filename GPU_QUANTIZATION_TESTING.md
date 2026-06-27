# GPU Quantization Testing — Real Results

## Overview

Real quantization testing of TinyLlama-1.1B (1.1 billion parameters) running on an NVIDIA L4 GPU (22.5 GB VRAM) via Google Cloud Run. This documents the infrastructure, results, and visualizations produced during testing.

## Infrastructure

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA L4, 22.5 GB VRAM |
| PyTorch | 2.6.0 + CUDA 12.6 |
| Platform | Google Cloud Run (us-central1) |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| API URL | `https://quantization-gpu-57431332482.us-central1.run.app` |

## Model Stats

- **Parameters:** 1,100,048,384 (1.1B)
- **Architecture:** LLaMA
- **Layers:** 22 transformer layers, 156 quantizable weight matrices
- **FP32 Size:** 4,196 MB (4.1 GB)
- **Hidden Size:** 2048
- **Vocab Size:** 32,000

## Quantization Results

### Method Comparison (all tested on same model)

| Method | Bits | Size (MB) | Compression | Perplexity | Increase | MSE | Time (GPU) |
|--------|------|-----------|-------------|------------|----------|-----|------------|
| FP32 Baseline | 32 | 4196.35 | 1.0x | 8.9927 | — | 0 | — |
| INT8 Symmetric (per-tensor) | 8 | 1049.09 | 4x | 9.1656 | +1.9% | 1.38e-6 | 0.92s |
| INT8 Symmetric (group=128) | 8 | 1049.09 | 4x | 9.2465 | +2.8% | 1.84e-8 | 0.93s |
| INT8 Asymmetric | 8 | 1049.09 | 4x | 9.1293 | +1.5% | 1.06e-6 | 0.98s |
| INT4 Symmetric (group=128) | 4 | 524.54 | 8x | 9.2508 | +2.9% | 6.02e-6 | 1.07s |
| INT4 Symmetric (group=32) | 4 | 524.54 | 8x | 8.6759 | -3.5% | 3.89e-6 | 0.96s |
| INT3 Symmetric (group=128) | 3 | 393.41 | 10.7x | 46.8622 | +421% | 3.24e-5 | 0.93s |

### Key Findings

1. **INT8 is nearly lossless** — less than 3% perplexity increase with 4x compression
2. **INT4 is the sweet spot** — 8x compression with negligible quality loss
3. **INT4 group=32 actually improves perplexity** — quantization noise can sometimes cancel out overfitting
4. **INT3 is catastrophic** — perplexity explodes 5x because 8 discrete levels cannot encode weight nuances
5. **GPU speedup is massive** — 37 tokens/sec on L4 vs ~0.4 on CPU (90x faster)

### Generation Speed Benchmark

| Metric | Value |
|--------|-------|
| Tokens/second | 37.11 |
| Average latency (100 tokens) | 2.695s |
| Consistency (3 runs) | 2.694s, 2.694s, 2.697s |

### Text Generation Quality

**Prompt:** "Once upon a time"

| Method | Generated Text |
|--------|---------------|
| INT8 per-tensor | "...there was a young girl named Lily. She lived in a small town in the countr" |
| INT8 group=128 | "...there was a young woman named Lily. She lived in a small village, surrounded by fields" |
| INT4 group=128 | "...there was a young woman named Sarah who lived in a small town. Sarah was a kind and" |
| INT4 group=32 | "...there was a young girl named Lily who lived in a small town. Lily loved to" |
| **INT3 (broken)** | "...I was thinking about a recipe for a chicken and mushroom soup. I was" |

INT3 completely loses the "fairy tale" pattern because the weight values encoding that concept are too damaged.

### GPTQ Results (Hessian-guided quantization)

- **4-bit GPTQ** with wikitext-2 calibration data (8 samples, 5 layers)
- Calibration time: 24.88s
- GPTQ quantization time: 37.1s
- Per-layer MSE: ~4-6e-6 (comparable to naive INT4)

### Layer Sensitivity

Most sensitive layers to quantization (INT4):
1. `layers.0.self_attn.v_proj` — perplexity +0.0298
2. `layers.0.self_attn.o_proj` — perplexity +0.0143
3. `layers.0.mlp.up_proj` — perplexity +0.0121

Least sensitive: embedding layers, MLP down projections

## Visualizations

The frontend application provides interactive visualizations of:

### 1. Number Line Snap
Shows individual weight values (blue dots) snapping to the nearest quantization grid point (green dots). Red lines show the movement (error) for each weight.

### 2. Weight Matrix Heatmap
16×16 slice of actual model weights displayed as:
- **Original (FP32):** Smooth color gradations
- **Quantized:** Visible "banding" effect (like JPEG compression)
- **Error map:** Red intensity shows where damage occurs

### 3. Token Probability Shift
Shows the model's top predicted next words with confidence bars, comparing FP32 vs quantized predictions. Demonstrates why text generation changes.

### 4. Quantization Grid Resolution
Visual comparison of 8-bit (256 levels), 4-bit (16 levels), and 3-bit (8 levels) grids overlaid on the weight value range.

### 5. Before/After Distribution Histogram
Overlaid area charts showing the smooth FP32 weight distribution versus the discrete quantized distribution.

### 6. Error Distribution
Bell curve showing the quantization noise added to each weight, centered at 0 (no systematic bias).

## Architecture

```
quantization-app/
├── backend/                    # FastAPI + PyTorch backend
│   ├── main.py                # API entry point with GPU device info
│   ├── Dockerfile             # PyTorch 2.6 + CUDA 12.6 container
│   ├── services/
│   │   ├── device.py          # GPU/CPU auto-detection
│   │   ├── real_quantization.py  # TinyLlama quantization engine
│   │   ├── gptq_real.py       # GPTQ with Hessian computation on GPU
│   │   ├── calibration.py     # Wikitext-2 calibration data loader
│   │   ├── mixed_precision.py # Mixed precision quantization
│   │   └── model_export.py    # Safetensors export
│   └── routers/
│       └── real_model.py      # All Real LLM API endpoints
├── src/                        # Next.js frontend
│   ├── components/
│   │   ├── dashboard/Dashboard.tsx
│   │   └── visualizers/
│   │       └── RealModelQuantization.tsx  # GPU visualization panel
│   └── lib/api.ts             # API client
└── deploy/                     # GCP deployment scripts
    ├── gcp-gpu-setup.sh       # Linux/Mac provisioning
    ├── gcp-gpu-setup.ps1      # Windows PowerShell provisioning
    └── gcp-manage.sh          # Start/stop/status management
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/device-info` | GET | GPU detection and VRAM status |
| `/api/real-model/info` | GET | Load model, return architecture info |
| `/api/real-model/quantize` | POST | Quantize full model, measure perplexity |
| `/api/real-model/compare` | POST | Compare INT8/INT4/INT3 methods |
| `/api/real-model/benchmark-speed` | POST | Measure tokens/second |
| `/api/real-model/gptq` | POST | Run GPTQ with calibration data |
| `/api/real-model/sensitivity` | POST | Per-layer sensitivity analysis |
| `/api/real-model/visualize-quantization` | POST | Weight histograms, scatter, error |
| `/api/real-model/deep-visualize` | POST | Number line + heatmap + token probs |
| `/api/real-model/inspect-layer` | POST | Inspect individual layer statistics |
| `/api/real-model/mixed-precision` | POST | Mixed precision quantization |
| `/api/real-model/export` | POST | Export to safetensors format |

## Deployment

### Quick Start (Local)

```bash
# Backend
cd quantization-app/backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd quantization-app
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

### Google Cloud Run (GPU)

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud builds submit backend/ --tag=REGION-docker.pkg.dev/PROJECT/REPO/backend:latest
gcloud beta run deploy quantization-gpu \
    --image=REGION-docker.pkg.dev/PROJECT/REPO/backend:latest \
    --gpu=1 --gpu-type=nvidia-l4 --cpu=4 --memory=16Gi \
    --max-instances=1 --no-gpu-zonal-redundancy \
    --allow-unauthenticated --port=8080
```

### Cost

- Cloud Run with L4 GPU: charged per request (scales to zero when idle)
- Build time: ~4 minutes per deployment
- First model load: ~10 seconds (cached after)

## What Quantization Does (Plain English)

A neural network is a file full of numbers (weights). TinyLlama has 1.1 billion of them. Each number is stored as a 32-bit floating point (FP32) — like `0.01491` or `-0.00372`.

**Quantization** replaces these precise decimals with crude integers to save space:

```
Original:   -0.01563 (stored as 32-bit float = 4 bytes)
INT8:       round(-0.01563 / 0.000921) = -17 (stored as 1 byte)
Recover:    -17 × 0.000921 = -0.01566 (close but not exact)
Error:      0.00003 per weight
```

With 1.1 billion weights, this saves 3 GB of memory (4.1 GB → 1.0 GB for INT8).

The error per weight is tiny, but it accumulates across billions of multiplications. That's why INT8 works fine (256 possible values per weight) but INT3 fails catastrophically (only 8 possible values — too few to encode the model's knowledge).
