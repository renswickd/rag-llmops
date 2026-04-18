from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_manager, get_session_registry, get_faiss_manager
from api.schemas.chat import ChatRequest, ChatResponse
from conversation.chat_manager import ChatManager
from ingestion.faiss_manager import FaissManager
from ingestion.data_ingestion import DataIngestion
from core.exceptions import RagAssistantException
from core.session_store import SessionRegistry
from core.logging_config import get_logger
from core.config import load_config
from pathlib import Path
import shutil

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])
config = load_config()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, chat_mgr: ChatManager = Depends(get_chat_manager)):
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
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
        )
        return ChatResponse(**result)
    except RagAssistantException as e:
        log.error("Chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e.error_message))


@router.get("/sessions")
def list_sessions(chat_mgr: ChatManager = Depends(get_chat_manager), session_registry: SessionRegistry = Depends(get_session_registry)):
    """List all active conversation session IDs."""
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")
    if session_registry is None:
        raise HTTPException(status_code=503, detail="Session registry not initialised.")
    
    try:
        sessions = [s["session_id"] for s in session_registry.list_sessions()]
        log.info("Listed sessions", count=len(sessions))
        return {"sessions": sessions}
    except Exception as e:
        log.error("Failed to list sessions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    chat_mgr: ChatManager = Depends(get_chat_manager),
    registry: SessionRegistry = Depends(get_session_registry),
    faiss_mgr: FaissManager = Depends(get_faiss_manager),
):
    """
    Fully delete a session:
    1. Remove from session registry
    2. Remove archived files from data/uploads/<session_id>/
    3. Rebuild the FAISS index from remaining sessions
    4. Clear conversation history from memory
    """
    if not registry.exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    try:
        # 1. Remove from registry
        registry.delete(session_id)

        # 2. Remove archived files
        uploads_dir = Path(config["data"]["data_dir"]) / "uploads"
        session_dir = uploads_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            log.info("Session files deleted", session_id=session_id, path=str(session_dir))

        # 3. Rebuild FAISS index from remaining sessions
        remaining_sessions = registry.list_sessions()
        faiss_mgr.clear()

        if remaining_sessions:
            all_chunks = []
            for session_meta in remaining_sessions:
                sid = session_meta["session_id"]
                sid_dir = uploads_dir / sid
                if not sid_dir.exists():
                    continue
                for file_path in sid_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt", ".md"}:
                        di = DataIngestion(
                            data_dir=sid_dir,
                            faiss_manager=faiss_mgr,
                            session_id=sid,
                        )
                        raw_docs = di.load_file(file_path)
                        chunks = di.chunk_documents(raw_docs)
                        all_chunks.extend(chunks)
            if all_chunks:
                faiss_mgr.create(all_chunks)
                log.info("FAISS index rebuilt", total_chunks=len(all_chunks), remaining_sessions=len(remaining_sessions))
        else:
            log.info("No remaining sessions — FAISS index is now empty")

        # 4. Clear in-memory chat history
        chat_mgr.clear_session(session_id)
        log.info("Chat history cleared", session_id=session_id)

        return {"message": f"Session '{session_id}' deleted.", "session_id": session_id}

    except Exception as e:
        log.error("Failed to delete session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))