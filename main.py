from utils.model_loader import ModelLoader


if __name__ == "__main__":
    # -----------------------------
    # Test conversational retrieval chain
    # -----------------------------
    from pathlib import Path
    from src.document_ingestion.faiss_manager import FaissManager
    from src.conversation.chat_manager import ChatManager
    from src.document_ingestion.retriever import Retriever
    from src.document_ingestion.data_ingestion import DataIngestion

    # faiss_manager = FaissManager(index_dir=Path("faiss_smoke_index"))

    # Sample documents already ingested - in folder data/sample_docs/
    faiss_manager = FaissManager(index_dir=Path("faiss_smoke_index"))
    # data_ingestion = DataIngestion(data_dir=Path("data/sample_docs"),
    #                                   faiss_manager=faiss_manager)
    # data_ingestion.ingest()  # Ingest sample documents into FAISS index
    
    retriever = Retriever(faiss_manager=faiss_manager, top_k=2)
    retriever.initialize()  # Load existing FAISS index
    
    chat_manager = ChatManager(retriever=retriever)
    
    sid = "smoke-session-1"
 
    q1 = "What does Context engineering"
    r1 = chat_manager.chat(q1, session_id=sid)
    print(f"\nQ: {q1}\nA: {r1['answer']}\nSources: {r1['sources']}\n")
 
    q2 = "How does it compare to traditional prompt engineering?"
    r2 = chat_manager.chat(q2, session_id=sid)
    print(f"Q: {q2}\nA: {r2['answer']}\nStandalone Q sent to retriever: {r2['standalone_q']}\n")
