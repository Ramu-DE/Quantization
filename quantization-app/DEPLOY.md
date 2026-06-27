# Deployment Guide - Quantization Visualizer

## Recommended Hardware

| Component | Minimum | Recommended (for Real LLM tab) |
|-----------|---------|-------------------------------|
| RAM | 8 GB | 32 GB+ |
| Disk | 10 GB free | 50 GB free |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | NVIDIA with 8GB+ VRAM (optional, for fast inference) |

The "Real LLM" tab downloads TinyLlama-1.1B (~2GB download, ~4.2GB in FP32 memory). All other tabs work fine on minimal hardware.

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | https://python.org/downloads |
| Node.js | 18+ | https://nodejs.org |
| uv (Python package manager) | latest | `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

## Quick Start (5 minutes)

```bash
# 1. Clone or copy the project
cd quantization-app

# 2. Install and start backend
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 &

# 3. Install and start frontend (in another terminal)
cd ..
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

---

## Step-by-Step Deployment

### 1. Backend (FastAPI + Python)

```bash
cd quantization-app/backend

# Install dependencies with uv
uv sync

# Test it works
uv run python -c "from services.quantizer import *; print('OK')"

# Start the server
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend will:
- Download ResNet50 weights (~100MB) on first request to `/api/weights/resnet50`
- Serve the API at http://localhost:8000
- Swagger docs at http://localhost:8000/docs

### 2. Frontend (Next.js)

```bash
cd quantization-app

# Install node dependencies
npm install

# Development mode
npm run dev

# OR production build
npm run build
npm run start
```

Frontend runs at http://localhost:3000.

---

## Production Deployment

### Option A: Run as Background Services (Linux/macOS)

```bash
# Backend
cd quantization-app/backend
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Frontend (production build)
cd ..
npm run build
nohup npm run start > frontend.log 2>&1 &
```

### Option B: Run as Background Services (Windows)

```powershell
# Backend (PowerShell)
cd quantization-app\backend
Start-Process -NoNewWindow -FilePath "uv" -ArgumentList "run","uvicorn","main:app","--host","0.0.0.0","--port","8000" -RedirectStandardOutput backend.log

# Frontend
cd ..
npm run build
Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run","start" -RedirectStandardOutput frontend.log
```

### Option C: Docker (recommended for production)

Create these files in the `quantization-app/` folder:

**Dockerfile.backend**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install uv && uv sync
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.frontend**
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000
CMD ["npm", "run", "start"]
```

**docker-compose.yml**
```yaml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
```

```bash
docker-compose up -d
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL (set this if backend is on different host/port) |

Create a `.env.local` file in `quantization-app/` to override:

```env
NEXT_PUBLIC_API_URL=http://your-backend-server:8000
```

### Accessing from Another Machine

If deploying on a server and accessing from another machine:

1. Set the backend to bind to `0.0.0.0` (already done in commands above)
2. Set `NEXT_PUBLIC_API_URL` to the server's IP/hostname:
   ```env
   NEXT_PUBLIC_API_URL=http://192.168.1.100:8000
   ```
3. Access the UI at `http://192.168.1.100:3000`

### CORS Configuration

The backend allows requests from `localhost:3000` by default. To allow other origins, edit `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Verify Deployment

```bash
# Check backend health
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# Check API docs
curl http://localhost:8000/docs
# Should return Swagger HTML

# Check frontend
curl -s http://localhost:3000 | grep "Quantization Visualizer"
# Should find the title
```

---

## Ports Summary

| Service | Port | URL |
|---------|------|-----|
| Backend API | 8000 | http://localhost:8000 |
| API Docs (Swagger) | 8000 | http://localhost:8000/docs |
| Frontend UI | 3000 | http://localhost:3000 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Install: `pip install uv` |
| Backend port in use | Kill existing: `lsof -ti:8000 \| xargs kill` (Linux) or `netstat -ano \| findstr :8000` (Windows) |
| Frontend port in use | Kill existing or use `npm run dev -- --port 3001` |
| CORS errors in browser | Update `allow_origins` in `backend/main.py` to include your frontend URL |
| ResNet50 download slow | First load takes time (~100MB model download). Subsequent loads use cache. |
| `ml-dtypes` import error | Ensure Python 3.11+. Run `uv sync` again. |
| Node version error | Requires Node 18+. Check with `node --version` |
