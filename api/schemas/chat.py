from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    session_id: str = Field(..., description="Identifies the conversation session")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    history_len: int
    standalone_q: str

class HistoryMessage(BaseModel):
    role: str          # "human" or "ai"
    content: str
    timestamp: str     # ISO-8601 UTC string

class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
