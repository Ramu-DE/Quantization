#!/bin/bash
# Start Quantization Visualizer - Linux/macOS
# Usage: ./start.sh

set -e

echo "========================================="
echo "  Quantization Visualizer - Starting"
echo "========================================="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found. Install Python 3.11+"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "ERROR: node not found. Install Node.js 18+"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; pip install uv; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start Backend
echo ""
echo "[1/2] Starting Backend (port 8000)..."
cd "$SCRIPT_DIR/backend"
uv sync --quiet
uv run uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "      Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "      Waiting for backend..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "      Backend ready!"
        break
    fi
    sleep 1
done

# Start Frontend
echo ""
echo "[2/2] Starting Frontend (port 3000)..."
cd "$SCRIPT_DIR"
if [ ! -d "node_modules" ]; then
    echo "      Installing dependencies..."
    npm install --silent
fi
npm run dev &
FRONTEND_PID=$!
echo "      Frontend PID: $FRONTEND_PID"

# Wait for frontend
sleep 5
echo ""
echo "========================================="
echo "  Ready!"
echo "  UI:       http://localhost:3000"
echo "  API:      http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop both services"

# Trap to kill both on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'; exit 0" SIGINT SIGTERM

# Wait
wait
