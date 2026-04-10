"""
Service initialization and FastAPI dependency providers.

All heavy objects (LLM, embeddings, FAISS, ChatManager) are created ONCE
at application startup via init_services(), then shared across every request
through FastAPI's Depends() system.

Why this matters:
- Loading an embedding model or LLM takes several seconds.
- If we created them per-request the API would be unusably slow.
- This pattern (create once, inject everywhere) is called Dependency Injection.
"""
from pathlib import Path
from typing import Optional

from core.config import load_config
from core.logging_config import get_logger
from utils.model_loader import ModelLoader
from ingestion.faiss_manager import FaissManager
from ingestion.retriever import Retriever
from conversation.chat_manager import ChatManager

log = get_logger(__name__)
config = load_config()

# Module-level singletons — set once at startup, reused for every request.
_model_loader: Optional[ModelLoader] = None
_faiss_manager: Optional[FaissManager] = None
_retriever: Optional[Retriever] = None
_chat_manager: Optional[ChatManager] = None


def init_services() -> None:
    """
    Called once when the FastAPI app starts up.
    Builds the full service chain: ModelLoader → FaissManager → Retriever → ChatManager.
    """
    global _model_loader, _faiss_manager, _retriever, _chat_manager

    log.info("Initializing services...")

    index_dir = Path(config["api"]["index_dir"])

    _model_loader = ModelLoader()

    _faiss_manager = FaissManager(index_dir=index_dir, model_loader=_model_loader)

    _retriever = Retriever(faiss_manager=_faiss_manager)

    # If an existing FAISS index is on disk, load it so the retriever is
    # ready to answer questions immediately without requiring an upload first.
    if _faiss_manager._exists():
        _faiss_manager.load()
        log.info("Existing FAISS index loaded at startup", index_dir=str(index_dir))
    else:
        log.warning("No existing FAISS index found — upload a document to get started", index_dir=str(index_dir))

    _chat_manager = ChatManager(retriever=_retriever)

    log.info("All services initialized successfully")


# ──────────────────────────────────────────────
# FastAPI dependency providers
# Each function below is passed to Depends() in
# the routers so FastAPI injects the right object.
# ──────────────────────────────────────────────

def get_chat_manager() -> ChatManager:
    return _chat_manager


def get_faiss_manager() -> FaissManager:
    return _faiss_manager


def get_retriever() -> Retriever:
    return _retriever
