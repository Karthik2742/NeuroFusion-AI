"""
NeuroFusion AI - FastAPI Main Application
AI-Powered MRI Enhancement & Intelligent Disease Localization

Entry point for the backend server.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.api.routes import router as api_router
from backend.api.websocket import progress_manager
from backend.configs import config


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
)
logger.add(
    "logs/neurofusion_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    logger.info("=" * 60)
    logger.info("  NeuroFusion AI - Starting Up")
    logger.info("  AI-Powered MRI Enhancement & Disease Localization")
    logger.info("=" * 60)

    # Create required directories
    dirs = [
        config.get("paths.dataset_dir", "./dataset"),
        config.get("paths.brain_dir", "./dataset/brain"),
        config.get("paths.weights_dir", "./weights"),
        config.get("paths.outputs_dir", "./outputs"),
        config.get("paths.reports_dir", "./reports"),
        config.get("paths.logs_dir", "./logs"),
        config.get("paths.temp_dir", "./temp"),
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            logger.info(f"  GPU: {gpu_name} ({gpu_mem} GB)")
        else:
            logger.warning("  GPU: Not available (CPU mode)")
    except Exception:
        logger.warning("  GPU: PyTorch CUDA not available")

    logger.info(f"  API: http://localhost:{config.get('app.port', 8000)}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("NeuroFusion AI - Shutting Down")


# Create FastAPI application
app = FastAPI(
    title="NeuroFusion AI",
    description=(
        "BRAIN TUMOR DETECTION using MONAI. "
        "Research-grade platform for Brain MRI analysis."
    ),
    version=config.get("app.version", "1.0.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
cors_origins = config.get("app.cors_origins", [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "*",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await progress_manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"WS received from {client_id}: {data}")
    except WebSocketDisconnect:
        await progress_manager.disconnect(client_id)


# Health check
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NeuroFusion AI",
        "version": config.get("app.version", "1.0.0"),
    }


# Root redirect
@app.get("/", tags=["System"])
async def root():
    """API root - service information."""
    return {
        "name": "NeuroFusion AI",
        "tagline": "AI-Powered MRI Enhancement & Intelligent Disease Localization",
        "version": config.get("app.version", "1.0.0"),
        "docs": "/docs",
        "api_prefix": "/api",
        "endpoints": {
            "upload": "POST /api/upload",
            "analyze": "POST /api/analyze",
            "preprocess": "POST /api/preprocess",
            "enhance": "POST /api/enhance",
            "evaluate": "POST /api/evaluate",
            "download": "GET /api/download/{file_id}",
            "gpu_status": "GET /api/gpu",
            "models": "GET /api/models/enhancement",
            "files": "GET /api/files",
            "websocket": "WS /ws/{client_id}",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.get("app.host", "0.0.0.0"),
        port=config.get("app.port", 8000),
        reload=config.get("app.debug", True),
        log_level="info",
    )
