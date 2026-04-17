from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_manager, get_session_registry
from api.schemas.chat import ChatRequest, ChatResponse
from conversation.chat_manager import ChatManager
from core.exceptions import RagAssistantException
from core.session_store import SessionRegistry
from core.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


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
def delete_session(session_id: str, chat_mgr: ChatManager = Depends(get_chat_manager)):
    """Clear the conversation history for a given session."""
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")
    try:
        cleared = chat_mgr.clear_session(session_id)
        if not cleared:
            log.warning("Attempted to clear non-existent session", session_id=session_id)
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return {"message": f"Session '{session_id}' cleared.", "session_id": session_id}
    except RagAssistantException as e:
        log.error("Failed to clear session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
