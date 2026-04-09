from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_manager
from api.schemas.chat import ChatRequest, ChatResponse
from src.conversation.chat_manager import ChatManager
from core.exceptions import RagAssistantException
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
def list_sessions(chat_mgr: ChatManager = Depends(get_chat_manager)):
    """List all active conversation session IDs."""
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")
    return {"sessions": chat_mgr.list_sessions()}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, chat_mgr: ChatManager = Depends(get_chat_manager)):
    """Clear the conversation history for a given session."""
    if chat_mgr is None:
        raise HTTPException(status_code=503, detail="Chat service not initialised.")
    cleared = chat_mgr.clear_session(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"message": f"Session '{session_id}' cleared.", "session_id": session_id}
