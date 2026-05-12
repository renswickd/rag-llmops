from typing import Optional
from datetime import datetime, timezone
import json
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from utils.model_loader import ModelLoader
from core.config import load_config
from core.logging_config import get_logger
from utils.file_handling import generate_session_id
from core.exceptions import RagAssistantException
from ingestion.retriever import Retriever
from conversation.prompt_builder import RAG_PROMPT, STANDALONE_PROMPT, format_docs

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
            LLM temperature.  Lower -> more deterministic.
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
        storage=None,
        session_id: Optional[str] = None,
    ):
        
        self.log = get_logger(__name__)
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.condense_question = condense_question
        self.max_history_turns = max_history_turns

        self._storage = storage
 
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
 
        history = self._get_or_create_history_session(session_id)
        chat_history: list[BaseMessage] = self._windowed_history(history)
 
        try:
            # 1. Condense follow-up into standalone query
            standalone_q = self._condense(question, chat_history)
 
            # 2. Retrieve relevant context
            docs = self.retriever.retrieve(standalone_q, top_k=top_k, session_id=session_id)
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
            self._persist_turn(session_id, question, answer)
 
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
    
    def get_history(self, session_id: str) -> list[dict]:
        if not self._storage:
            return []
        lines = self._storage.read_history(session_id)
        return [json.loads(line) for line in lines]
    
    def clear_session(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        if existed:
            del self._sessions[session_id]
            self.log.info("Session cleared", session_id=session_id)
        if self._storage:
            self._storage.delete_history(session_id)
        return existed
        
    # ---------------
    # Helper methods
    # ---------------
    def _get_or_create_history_session(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
            self.log.info("New history session created", session_id=session_id)
        return self._sessions[session_id]
    
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


    def _persist_turn(self, session_id: str, human_content: str, ai_content: str) -> None:
        if not self._storage:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._storage.append_history(session_id,json.dumps({"role": "human", "content": human_content, "timestamp": now}))
        self._storage.append_history(session_id,json.dumps({"role": "ai", "content": ai_content, "timestamp": now}))


    def _load_persisted_history(self, known_session_ids: list[str] | None = None) -> None:
        """
        Load persisted history into memory at startup.

        known_session_ids: list of session IDs from the registry.  If provided, only these sessions are loaded.  If None, no history is loaded (no-op).
        """
        if not self._storage or not known_session_ids:
            return
        loaded = 0
        for session_id in known_session_ids:
            lines = self._storage.read_history(session_id)
            if not lines:
                continue
            history = InMemoryChatMessageHistory()
            for line in lines:
                msg = json.loads(line)
                if msg["role"] == "human":
                    history.add_message(HumanMessage(content=msg["content"]))
                else:
                    history.add_message(AIMessage(content=msg["content"]))
            self._sessions[session_id] = history
            loaded += 1
        self.log.info("Persisted history loaded", sessions_loaded=loaded)

    # ----------
    # Chains
    # ----------
        
    def _build_answer_chain(self):
        """
        LCEL chain:  prompt -> LLM -> string output
        Inputs: {context, chat_history, question}
        """
        return RAG_PROMPT | self._llm | StrOutputParser()
 
    def _build_condense_chain(self):
        """
        LCEL chain:  prompt -> LLM -> string output
        Inputs: {chat_history, question}
        """
        return STANDALONE_PROMPT | self._llm | StrOutputParser()

if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path
    from langchain_core.documents import Document
    from ingestion.faiss_manager import FaissManager
    from core.storage import LocalStorageBackend

    sample_docs = [
        Document(page_content="FAISS enables fast vector similarity search at scale.", metadata={"source": "faiss.txt"}),
        Document(page_content="RAG combines retrieval with generation for accurate answers.", metadata={"source": "rag.txt"}),
        Document(page_content="Groq provides ultra-fast LLM inference via dedicated hardware.", metadata={"source": "groq.txt"}),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalStorageBackend(data_dir=Path(tmp))
        manager = FaissManager(index_dir=Path(tmp) / "faiss")
        retriever = Retriever(faiss_manager=manager, top_k=2)
        retriever.initialize(docs=sample_docs)

        cm = ChatManager(retriever=retriever, storage=backend)
        sid = "smoke-storage-1"

        # 1. chat() -> _persist_turn() -> storage
        q1 = "What is FAISS?"
        r1 = cm.chat(q1, session_id=sid)
        raw_lines = backend.read_history(sid)
        assert len(raw_lines) == 2, f"Expected 2 history lines after 1 turn, got {len(raw_lines)}"
        assert json.loads(raw_lines[0])["role"] == "human"
        assert json.loads(raw_lines[1])["role"] == "ai"
        print(f"  [1] _persist_turn writes both messages to storage  ({len(raw_lines)} lines)")

        # Second turn appends — does not overwrite
        q2 = "How does it compare to RAG?"
        cm.chat(q2, session_id=sid)
        raw_lines = backend.read_history(sid)
        assert len(raw_lines) == 4, f"Expected 4 lines after 2 turns, got {len(raw_lines)}"
        print(f"  [2] second turn appended correctly              ({len(raw_lines)} lines)")

        # 2. get_history() reads back from storage
        history = cm.get_history(sid)
        assert len(history) == 4
        assert history[0]["content"] == q1
        assert history[0]["role"] == "human"
        print(f"  [3] get_history() reads {len(history)} messages from storage")

        # 3. _load_persisted_history() restores sessions in a fresh ChatManager
        cm2 = ChatManager(retriever=retriever, storage=backend)
        assert sid not in cm2._sessions, "Fresh ChatManager must start with empty sessions"
        cm2._load_persisted_history(known_session_ids=[sid])
        assert sid in cm2._sessions, "_load_persisted_history must populate _sessions"
        assert len(cm2._sessions[sid].messages) == 4
        print(f"  [4] _load_persisted_history restored {len(cm2._sessions[sid].messages)} messages")

        # 4. clear_session() removes from memory AND deletes from storage
        existed = cm.clear_session(sid)
        assert existed, "clear_session must return True when session existed"
        assert sid not in cm._sessions, "clear_session must remove session from _sessions"
        assert backend.read_history(sid) == [], "clear_session must delete history from storage"
        print("  [5] clear_session removed session from memory and storage")

        # 5. clear_session() on unknown session returns False and does not crash
        result = cm.clear_session("nonexistent-session")
        assert result is False
        print("  [6] clear_session on unknown session returns False without crashing")

    print("=== Step 6 storage smoke PASSED ===\n")

    # ── LLM pipeline smoke test (requires GROQ_API_KEY in .env) ──────────────
    if not __import__("os").getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — skipping LLM pipeline smoke test")
        sys.exit(0)

    print("=== LLM pipeline smoke (storage + retrieval) ===")
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalStorageBackend(data_dir=Path(tmp))
        manager = FaissManager(index_dir=Path(tmp) / "faiss")
        retriever = Retriever(faiss_manager=manager, top_k=2)
        retriever.initialize(docs=sample_docs)

        cm = ChatManager(retriever=retriever, storage=backend)
        sid = "smoke-llm-1"

        r1 = cm.chat("What is FAISS?", session_id=sid)
        print(f"\nQ: What is FAISS?\nA: {r1['answer']}\nSources: {r1['sources']}")
        assert backend.read_history(sid), "History must be persisted after chat()"

        r2 = cm.chat("How does it compare to RAG?", session_id=sid)
        print(f"\nQ: How does it compare to RAG?\nA: {r2['answer']}\nStandalone Q: {r2['standalone_q']}")
        assert len(backend.read_history(sid)) == 4, "Two turns must produce 4 history lines"

    print("\n=== LLM pipeline smoke PASSED ===")