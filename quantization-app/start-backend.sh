#!/bin/bash
# Start the FastAPI backend

export PATH="/c/Users/Administrator/.local/bin:$PATH"
cd backend
echo "Starting FastAPI backend on port 8000..."
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
