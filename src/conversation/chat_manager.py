from typing import Optional
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from utils.model_loader import ModelLoader
from core.config import load_config
from core.logging_config import get_logger
from utils.file_handling import generate_session_id
from core.exceptions import RagAssistantException
from src.document_ingestion.retriever import Retriever
from src.conversation.prompt_builder import RAG_PROMPT, STANDALONE_PROMPT, format_docs

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
        self.session_id = session_id or generate_session_id("session")

        self._llm = ModelLoader().load_llm()
        self._answer_chain = self._build_answer_chain()
        self._condense_chain = self._build_condense_chain()
 
        self.log.info("ChatManager initialized", model=self.model_name, condense_question=self.condense_question, max_history_turns=self.max_history_turns)
        
    def chat(self, question: str, session_id: str, top_k: Optional[int] = None) -> dict:
        """
        Process one conversational turn.
 
        Args:
            question:   Raw user message.
            session_id: Identifies the conversation session.  A new session
                        is created automatically on first use.
            top_k:      Override retriever top_k for this turn only.
 
        Returns:
            dict with keys:
              - `answer`        (str)  : LLM-generated answer.
              - `sources`       (list) : list of source metadata dicts.
              - `session_id`    (str)  : echoed session_id.
              - `history_len`   (int)  : number of messages in history after this turn.
              - `standalone_q`  (str)  : condensed query sent to retriever
                                          (equals `question` if condensation is disabled or it's the first turn).
        """
        if not question or not question.strip():
            raise RagAssistantException("Question must not be empty.")
 
        history = self._get_or_create_history_session()
        chat_history: list[BaseMessage] = self._windowed_history(history)
 
        try:
            # 1. Condense follow-up into standalone query
            standalone_q = self._condense(question, chat_history)
 
            # 2. Retrieve relevant context
            docs = self.retriever.retrieve(standalone_q, top_k=top_k)
            context = format_docs(docs)
 
            # 3. Generate grounded answer
            answer = self._answer_chain.invoke(
                {
                    "context": context,
                    "chat_history": chat_history,
                    "question": question,
                }
            )
 
            # 4. Persist turn in session history
            history.add_message(HumanMessage(content=question))
            history.add_message(AIMessage(content=answer))
 
            sources = [
                {k: v for k, v in doc.metadata.items()}
                for doc in docs
            ]
 
            self.log.info(
                "Chat turn completed", session_id=session_id, standalone_q=standalone_q[:80], num_sources=len(sources), history_len=len(history.messages))
 
            return {
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "history_len": len(history.messages),
                "standalone_q": standalone_q,
            }
 
        except Exception as e:
            self.log.error("Chat turn failed", session_id=session_id, error=str(e))
            raise RagAssistantException("Chat turn failed", e) from e
    
    def clear_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self.log.info("Session cleared", session_id=session_id)
            return True
        return False
 
    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())
        
    # ---------------
    # Helper methods
    # ---------------
    def _get_or_create_history_session(self) -> InMemoryChatMessageHistory:
        if self.session_id not in self._sessions:
            self._sessions[self.session_id] = InMemoryChatMessageHistory()
            self.log.info("New history session created", session_id=self.session_id)
        return self._sessions[self.session_id]
    
    def _windowed_history(self, history: InMemoryChatMessageHistory ) -> list[BaseMessage]:
        """
        Return at most `max_history_turns` human+AI pairs from the tail of the history.  0 means return everything.
        """
        msgs = history.messages
        if self.max_history_turns == 0 or len(msgs) == 0:
            return list(msgs)

        cutoff = self.max_history_turns * 2
        return list(msgs[-cutoff:])
    
    def _condense(self, question: str, chat_history: list[BaseMessage]) -> str:
        """
        Rephrase *question* into a standalone query using conversation history.
        Returns *question* unchanged when condensation is off or history is empty.
        """
        if not self.condense_question or not chat_history:
            return question
 
        try:
            standalone = self._condense_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
            self.log.debug("Question condensed", original=question[:80],condensed=standalone[:80])
            return standalone
        
        except Exception as e:
            self.log.warning("Question condensation failed — using raw question", error=str(e))
            return question
        
    # ----------
    # Chains
    # ----------
        
    def _build_answer_chain(self):
        """
        LCEL chain:  prompt → LLM → string output
        Inputs: {context, chat_history, question}
        """
        return RAG_PROMPT | self._llm | StrOutputParser()
 
    def _build_condense_chain(self):
        """
        LCEL chain:  prompt → LLM → string output
        Inputs: {chat_history, question}
        """
        return STANDALONE_PROMPT | self._llm | StrOutputParser()

if __name__ == "__main__":
    # CMObj = ChatManager(retriever=None)

    from pathlib import Path
    from langchain_core.documents import Document
    from src.document_ingestion.faiss_manager import FaissManager
 
    sample_docs = [
        Document(page_content="FAISS enables fast vector similarity search at scale.", metadata={"source": "faiss.txt"}),
        Document(page_content="RAG combines retrieval with generation for accurate answers.", metadata={"source": "rag.txt"}),
        Document(page_content="Groq provides ultra-fast LLM inference via dedicated hardware.", metadata={"source": "groq.txt"}),
    ]
 
    manager = FaissManager(index_dir=Path("faiss_smoke_index"))
    retriever = Retriever(faiss_manager=manager, top_k=2)
    retriever.initialize(docs=sample_docs)
 
    cm = ChatManager(retriever=retriever)
    sid = "smoke-session-1"
 
    q1 = "What is FAISS?"
    r1 = cm.chat(q1, session_id=sid)
    print(f"\nQ: {q1}\nA: {r1['answer']}\nSources: {r1['sources']}\n")
 
    q2 = "How does it compare to RAG?"
    r2 = cm.chat(q2, session_id=sid)
    print(f"Q: {q2}\nA: {r2['answer']}\nStandalone Q sent to retriever: {r2['standalone_q']}\n")    
