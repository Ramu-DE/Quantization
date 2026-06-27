# Multi-stage build: Backend + Frontend in one container
# Optimized for Cloud Run with GPU (NVIDIA L4)

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY package*.json ./
RUN npm ci --production=false
COPY src/ ./src/
COPY next.config.ts tsconfig.json postcss.config.mjs tailwind.config.ts ./
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# Stage 2: Final image with Python + Node runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and Node.js 20
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    nodejs npm curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Setup backend
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock* ./
RUN uv sync --no-dev 2>/dev/null || uv sync
COPY backend/ ./

# Setup frontend
WORKDIR /app/frontend
COPY --from=frontend-builder /app/frontend/.next ./.next
COPY --from=frontend-builder /app/frontend/node_modules ./node_modules
COPY package.json ./

# Setup nginx-like routing with a simple Python proxy
WORKDIR /app
COPY deploy/start-server.py ./

# Pre-download the model during build (cached in image)
RUN uv run --directory /app/backend python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading TinyLlama-1.1B...')
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('Model cached in image!')
"

EXPOSE 8080

CMD ["python3.11", "/app/start-server.py"]
