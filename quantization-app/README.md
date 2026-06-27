# AI Model Quantization Visualizer

Interactive web application for learning and exploring AI model quantization techniques — from basic concepts to advanced LLM methods.

## Features (11 Interactive Tabs)

| Tab | Content |
|-----|---------|
| **Formats** | Bit layouts for FP32, FP16, BFloat16, INT8, INT4 with visual diagrams |
| **Weights** | Load ResNet50 FC layer weights, histogram, CDF, outlier/threshold analysis |
| **Steps** | Step-by-step symmetric quantization walkthrough with reversibility indicators |
| **Compare** | Side-by-side: Symmetric, Asymmetric, Clipped INT8, BFloat16 with MSE/RSS |
| **Errors** | Error distribution histograms for all methods (uniform vs concentrated) |
| **Memory** | Memory usage comparison charts across formats (4x-8x savings) |
| **Playground** | Real-time sliders: value, bit-width, clip range with live quantization |
| **Advanced** | FP8 (E4M3/E5M2), GPTQ simulation with Hessian compensation, SmoothQuant |
| **Guide** | Decision wizard: model size + hardware + tolerance → method recommendation |
| **Benchmarks** | LLaMA-2-7B method comparison + live quantization benchmarks |
| **Hardware** | GPU throughput comparison (T4, A100, H100, RTX 4090, MI300X) |

## Quick Start

```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

Or manually:

```bash
# Backend (port 8000)
cd backend && uv sync && uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (port 3000)
npm install && npm run dev
```

**UI:** http://localhost:3000  
**API Docs:** http://localhost:8000/docs

## Tech Stack

- **Backend:** FastAPI, NumPy, PyTorch, ml-dtypes
- **Frontend:** Next.js 15, React 19, Recharts, Radix UI, Tailwind CSS, TanStack Query
- **State:** Zustand

## API Endpoints (20+)

### Core Quantization
- `POST /api/quantize/symmetric` — Symmetric INT8
- `POST /api/quantize/asymmetric` — Asymmetric INT8
- `POST /api/quantize/clipped` — Clipped INT8
- `POST /api/quantize/bfloat16` — BFloat16 casting
- `POST /api/quantize/steps` — Step-by-step walkthrough

### Analysis
- `POST /api/quantize/error-distribution` — Error histograms per method
- `POST /api/quantize/memory` — Memory comparison across formats
- `POST /api/weights/distribution` — Weight histogram and CDF
- `POST /api/weights/outliers` — Outlier threshold analysis
- `GET /api/weights/resnet50` — Load real ResNet50 weights

### Advanced Methods
- `POST /api/fp8/convert` — FP8 E4M3/E5M2 simulation
- `POST /api/gptq/simulate` — GPTQ with Hessian compensation
- `POST /api/smoothquant/simulate` — SmoothQuant transformation

### Decision and Benchmarks
- `POST /api/decision-guide` — Method recommendation engine
- `GET /api/benchmarks/methods` — LLaMA-2-7B benchmark data
- `POST /api/benchmarks/live` — Real-time quantization timing
- `GET /api/hardware/comparison` — GPU architecture specs
- `GET /api/formats` — Number format information

## Documentation

See `docs/` folder for comprehensive reference (160 KB, 11 files):
- Fundamentals, PTQ, QAT, Granularity
- NVIDIA, Microsoft, Google ecosystems
- LLM methods (GPTQ, AWQ, SmoothQuant, GGUF, etc.)
- Hardware formats (FP8, MX, BFP)
- Evaluation metrics and decision frameworks

## Deployment

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions including Docker.
