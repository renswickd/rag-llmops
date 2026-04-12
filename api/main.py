"""
FastAPI application entry point.

Start the server with:
    uvicorn api.main:app --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import init_services
from api.routers import health, documents, chat
from core.config import load_config
from core.logging_config import setup_logging, get_logger

config = load_config()
setup_logging(config)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at startup; code after `yield` runs at shutdown.
    This is the modern FastAPI way to handle startup/shutdown events.
    """
    log.info("Starting up RAG LLMOps API...")
    init_services()
    yield
    log.info("Shutting down RAG LLMOps API.")


app = FastAPI(
    title=config["app"]["name"],
    description=config["app"]["description"],
    version=config["app"]["version"],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers — all endpoints live under /api/v1/
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

# Serve built frontend from FastAPI in production (optional, for single-container deploy)
_frontend_dist = Path("frontend/dist")
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
