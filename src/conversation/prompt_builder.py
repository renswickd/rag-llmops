
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core.logging_config import get_logger
 
log = get_logger(__name__)

# ----------------------------------------
# # System message — RAG answer generation
# ----------------------------------------

_RAG_SYSTEM = """You are a knowledgeable, precise, and helpful AI assistant. Your job is to answer the user's question using ONLY the context passages provided below.
 
Rules:
1. Base your answer strictly on the provided context. Do NOT invent facts.
2. If the context does not contain enough information, say so clearly rather than guessing.
3. Cite the source of each relevant piece of information using the document's "source" metadata when available (e.g. "According to intro.txt, ...").
4. Be concise but complete. Use bullet points or short paragraphs as appropriate.
5. Maintain conversational continuity — the chat history is provided so you can resolve pronouns and follow-up references correctly.
 
---
CONTEXT:
{context}
---
"""

# -----------------------------------------------
# System message — standalone question rephrasing
# -----------------------------------------------
_STANDALONE_SYSTEM = """Given the conversation history and a follow-up question, \
rewrite the follow-up as a fully self-contained question that includes all \
necessary context from the history.
 
Rules:
- Output ONLY the rewritten question. No preamble, no explanation.
- Preserve the original intent and language of the user.
- If the follow-up is already self-contained, return it unchanged.
"""


# ----------------------
# Public prompt objects
# ----------------------
 
RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _RAG_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
"""
Prompt for answer generation.
 
Expected input keys:
  - context      : str  — concatenated retrieved document passages
  - chat_history : list — LangChain BaseMessage objects (may be empty)
  - question     : str  — current user question
"""
 
STANDALONE_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _STANDALONE_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
"""
Prompt for question condensation.
 
Expected input keys:
  - chat_history : list — LangChain BaseMessage objects (may be empty)
  - question     : str  — raw follow-up question from the user
"""
 
 
def format_docs(docs) -> str:
    """
    Serialize a list of LangChain Documents into a single context string
    suitable for injection into RAG_PROMPT.
 
    Each passage is prefixed with its source metadata (if present) so the
    LLM can cite it naturally.
 
    Args:
        docs: List[Document]
 
    Returns:
        Newline-separated string of passages.
    """
    if not docs:
        return "No relevant context found."
 
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"document-{i}")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content.strip()}")
 
    formatted = "\n\n".join(parts)
    log.debug("Context formatted", num_docs=len(docs), total_chars=len(formatted))
    return formatted