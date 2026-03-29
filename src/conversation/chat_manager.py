import os
from typing import Optional
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory

from utils.model_loader import ModelLoader
from core.config import load_config
from core.logging_config import get_logger
from utils.file_handling import generate_session_id
from core.exceptions import RagAssistantException
from src.document_ingestion.retriever import Retriever

load_dotenv()
config = load_config()

class ChatManager:
    """
    Stateful conversational manager for the RAG assistant.
 
    Args:
        retriever (Retriever):
            An *initialised* Retriever instance (vector store already loaded).
        model_name (str):
            Groq model identifier.  Defaults to config value.
        temperature (float):
            LLM temperature.  Lower → more deterministic.
        max_tokens (int):
            Maximum tokens in the LLM response.
        condense_question (bool):
            When True (default), a first LLM call rephrases follow-up
            questions before retrieval.  Disable for single-turn usage
            to save latency.
        max_history_turns (int):
            Maximum number of human+AI turn pairs retained per session.
            Older turns are dropped (sliding window).  0 = unlimited.
    """
 
    def __init__(
        self,
        retriever: Retriever,
        model_name: str = config["llm"]["groq"]["model_name"],
        temperature: float = config["llm"]["groq"]["temperature"],
        max_tokens: int = config["llm"]["groq"]["max_output_tokens"],
        condense_question: bool = config["llm"]["groq"].get("condense_question", True),
        max_history_turns: int = config["llm"]["groq"].get("max_history_turns", 10),
        session_id: Optional[str] = None,
    ):
        
        self.log = get_logger(__name__)
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.condense_question = condense_question
        self.max_history_turns = max_history_turns
 
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}
        
        self._llm = ModelLoader().load_llm()
        self.session_id = session_id or generate_session_id("session")
 
        self.log.info("ChatManager initialized", model=self.model_name, condense_question=self.condense_question, max_history_turns=self.max_history_turns)
        
        
        
    # ---------------
    # Helper methods
    # ---------------
    def _get_or_create_history_session(self) -> InMemoryChatMessageHistory:
        if self.session_id not in self._sessions:
            self._sessions[self.session_id] = InMemoryChatMessageHistory()
            self.log.info("New history session created", session_id=self.session_id)
        return self._sessions[self.session_id]
        

if __name__ == "__main__":
    CMObj = ChatManager(retriever=None)
    
