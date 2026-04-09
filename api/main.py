"""
FastAPI application entry point.

Start the server with:
    uvicorn api.main:app --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
    title="RAG LLMOps API",
    description="A document Q&A API powered by Retrieval-Augmented Generation.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers — all endpoints live under /api/v1/
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
