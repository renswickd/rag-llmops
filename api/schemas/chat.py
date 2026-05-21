from pydantic import BaseModel, Field
from typing import Optional

# ---> Chat Request
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="The user's question")
    session_id: str = Field(..., description="Identifies the conversation session")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve")

# ---> Chat Response
class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    history_len: int
    standalone_q: str

# ---> History Response
class HistoryMessage(BaseModel):
    role: str          # "human" or "ai"
    content: str
    timestamp: str     # ISO-8601 UTC string

class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]

# ---> Session List Response
class SessionDocument(BaseModel):
    file_name: str
    chunks_created: int

class SessionMetadata(BaseModel):
    session_id: str
    created_at: str          # ISO-8601 UTC string from registry
    documents: list[SessionDocument]

class SessionListResponse(BaseModel):
    sessions: list[SessionMetadata]
