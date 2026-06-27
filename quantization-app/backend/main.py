"""
FastAPI Backend for Quantization Visualizer
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import quantization, weights, formats, advanced, real_model

app = FastAPI(
    title="Quantization API",
    description="Backend API for AI Model Quantization Visualization",
    version="0.1.0"
)

# CORS middleware for Next.js frontend (allows any origin for remote GPU deployment)
import os
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_cors_origins = ["*"] if _cors_env.strip() == "*" else [o.strip() for o in _cors_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_env.strip() != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(quantization.router, prefix="/api", tags=["quantization"])
app.include_router(weights.router, prefix="/api", tags=["weights"])
app.include_router(formats.router, prefix="/api", tags=["formats"])
app.include_router(advanced.router, prefix="/api", tags=["advanced"])
app.include_router(real_model.router, prefix="/api", tags=["real-model"])


@app.get("/")
async def root():
    return {
        "message": "Quantization API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/device-info")
async def device_info():
    """Return GPU/CPU device information for diagnostics."""
    from services.device import get_device_info
    return get_device_info()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
