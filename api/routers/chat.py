import tempfile

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from api.limiter import limiter
from api.dependencies import get_chat_manager, get_session_registry, get_faiss_manager, get_storage
from api.schemas.chat import ChatRequest, ChatResponse, HistoryResponse, SessionListResponse, SessionMetadata
from conversation.chat_manager import ChatManager
from ingestion.faiss_manager import FaissManager
from ingestion.data_ingestion import DataIngestion
from core.storage import StorageBackend
from core.exceptions import RagAssistantException
from core.session_store import SessionRegistry
from core.logging_config import get_logger
from core.config import load_config
from pathlib import Path
import shutil

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])
config = load_config()

_chat_rpm = config.get("rate_limiting", {}).get("chat_rpm", 10)


@router.post("", response_model=ChatResponse)
@limiter.limit(f"{_chat_rpm}/minute")
def chat(
    request: Request,
    body: ChatRequest = Body(...),
    chat_mgr: ChatManager = Depends(get_chat_manager),
):
    """
    Send a question and get an answer grounded in your uploaded documents.

    - Use the same `session_id` across turns to maintain conversation history.
    - The API automatically rephrases follow-up questions before retrieval.
    """
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")

    if chat_mgr.retriever.faiss_manager.vs is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Upload a document first via POST /api/v1/documents/upload.",
        )

    try:
        result = chat_mgr.chat(
            question=body.question,
            session_id=body.session_id,
            top_k=body.top_k,
        )
        return ChatResponse(**result)
    except RagAssistantException as e:
        log.error("Chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e.error_message))


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(chat_mgr: ChatManager = Depends(get_chat_manager), session_registry: SessionRegistry = Depends(get_session_registry)):
    """List all sessions with metadata (session_id, created_at, documents)."""
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")
    if session_registry is None:
        raise HTTPException(status_code=503, detail="Session registry not initialised.")
    
    try:
        raw = session_registry.list_sessions()   # already sorted newest-first
        sessions = [SessionMetadata(**s) for s in raw]
        log.info("Listed sessions", count=len(sessions))
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        log.error("Failed to list sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    chat_mgr: ChatManager = Depends(get_chat_manager),
    registry: SessionRegistry = Depends(get_session_registry),
    faiss_mgr: FaissManager = Depends(get_faiss_manager),
    storage: StorageBackend = Depends(get_storage),
):
    """
    Fully delete a session:
    1. Remove from session registry
    2. Remove archived files from data/uploads/<session_id>/
    3. Rebuild the FAISS index from remaining sessions from Azure Files mount point
    4. Clear conversation history from memory
    """
    if chat_mgr is None or registry is None or faiss_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")

    if not registry.exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    try:
        # 1. Remove from registry
        registry.delete(session_id)

        # 2. Remove archived files
        storage.delete_session_files(session_id)
        log.info("Session files deleted from storage", session_id=session_id)

        # 3. Rebuild FAISS index from remaining sessions
        remaining_sessions = registry.list_sessions()
        faiss_mgr.clear()

        if remaining_sessions:
            all_chunks = []
            for session_meta in remaining_sessions:
                sid = session_meta["session_id"]
                filenames = storage.list_session_files(sid)
                for filename in filenames:
                    suffix = Path(filename).suffix.lower()
                    if suffix not in {".pdf", ".txt", ".md"}:
                        continue
                    # Download the file to a temp location for DataIngestion
                    file_bytes = storage.read_file(sid, filename)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = Path(tmp.name)
                    try:
                        di = DataIngestion(
                            data_dir=tmp_path.parent,
                            faiss_manager=faiss_mgr,
                            session_id=sid,
                        )
                        raw_docs = di.load_file(tmp_path)
                        chunks = di.chunk_documents(raw_docs)
                        all_chunks.extend(chunks)
                    finally:
                        tmp_path.unlink(missing_ok=True)

            if all_chunks:
                faiss_mgr.create(all_chunks)
                log.info("FAISS index rebuilt", total_chunks=len(all_chunks),
                         remaining_sessions=len(remaining_sessions))
        else:
            log.info("No remaining sessions — FAISS index is now empty")

        # 4. Clear in-memory chat history
        chat_mgr.clear_session(session_id)
        log.info("Chat history cleared", session_id=session_id)

        return {"message": f"Session '{session_id}' deleted.", "session_id": session_id}

    except Exception as e:
        log.error("Failed to delete session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_session_history(
    session_id: str,
    chat_mgr: ChatManager = Depends(get_chat_manager),
):
    """
    Return the stored conversation history for one session.
    Used by the frontend to re-hydrate messages after a page refresh.
    Returns 200 with an empty list if the session has no history yet.
    """
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")
    try:
        messages = chat_mgr.get_history(session_id)
        return HistoryResponse(session_id=session_id, messages=messages)
    except Exception as e:
        log.error("Failed to get history", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))