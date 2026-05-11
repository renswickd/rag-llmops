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
import os
from pathlib import Path
from typing import Optional

from core.config import load_config
from core.logging_config import get_logger
from core.session_store import SessionRegistry
from utils.model_loader import ModelLoader
from ingestion.faiss_manager import FaissManager
from ingestion.retriever import Retriever
from conversation.chat_manager import ChatManager
from core.storage import StorageBackend, create_storage_backend

log = get_logger(__name__)
config = load_config()

# Module-level singletons — set once at startup, reused for every request.
_model_loader: Optional[ModelLoader] = None
_faiss_manager: Optional[FaissManager] = None
_retriever: Optional[Retriever] = None
_chat_manager: Optional[ChatManager] = None
_session_registry: Optional[SessionRegistry] = None
_storage: Optional[StorageBackend] = None

def init_services() -> None:
    """
    Called once when the FastAPI app starts up.
    Builds the full service chain: ModelLoader → FaissManager → Retriever → ChatManager.
    """
    global _model_loader, _faiss_manager, _retriever, _chat_manager, _session_registry, _storage

    log.info("Initializing services...")

    index_dir = Path(config["api"]["index_dir"])
    data_dir = Path(config["data"]["data_dir"])

    configured_backend = config.get("storage", {}).get("backend", "local")
    app_env = os.environ.get("APP_ENVIRONMENT", "dev")
    resolved_backend = "azure_blob" if app_env == "production" else configured_backend

    _storage = create_storage_backend(backend=resolved_backend, data_dir=data_dir)
    log.info("Storage backend initialised", configured_backend=configured_backend, resolved_backend=resolved_backend, app_environment=app_env, data_dir=str(data_dir))

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

    _chat_manager = ChatManager(retriever=_retriever, storage=_storage)

    _session_registry = SessionRegistry(storage=_storage)

    known_ids = [session["session_id"] for session in _session_registry.list_sessions()]
    _chat_manager._load_persisted_history(known_session_ids=known_ids)

    log.info("All services initialized successfully")


# ──────────────────
# FastAPI dependency providers
# Each function below is passed to Depends() in the routers so FastAPI injects the right object.
# ──────────────────

def get_chat_manager() -> ChatManager:
    return _chat_manager

def get_faiss_manager() -> FaissManager:
    return _faiss_manager

def get_retriever() -> Retriever:
    return _retriever

def get_session_registry() -> SessionRegistry:
    return _session_registry

def get_storage() -> StorageBackend:
    return _storage