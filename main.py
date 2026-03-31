from utils.model_loader import ModelLoader


if __name__ == "__main__":
    # -----------------------------
    # Test conversational retrieval chain
    # -----------------------------
    from pathlib import Path
    from src.document_ingestion.faiss_manager import FaissManager
    from src.conversation.chat_manager import ChatManager
    from src.document_ingestion.retriever import Retriever
    from src.conversation.prompt_builder import RAG_PROMPT, STANDALONE_PROMPT, format_docs

    faiss_manager = FaissManager(index_dir=Path("faiss_smoke_index"))

    # Sample documents already ingested - in folder data/sample_docs/
    manager = FaissManager(index_dir=Path("faiss_smoke_index"))
    retriever = Retriever(faiss_manager=manager, top_k=2)
    retriever.initialize()  # Load existing FAISS index
    chat_manager = ChatManager(retriever=retriever)


    question = "What does Red-Tapism mean in the context of government bureaucracy?"
    # chat_history = []  # No prior conversation for this test
    
    sid = "smoke-session-1"
 
    q1 = "What does Red-Tapism mean in the context of government?"
    r1 = chat_manager.chat(q1, session_id=sid)
    print(f"\nQ: {q1}\nA: {r1['answer']}\nSources: {r1['sources']}\n")
 
    q2 = "How does it compare bureaucratic procedure?"
    r2 = chat_manager.chat(q2, session_id=sid)
    print(f"Q: {q2}\nA: {r2['answer']}\nStandalone Q sent to retriever: {r2['standalone_q']}\n")
