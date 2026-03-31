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
    chat_manager = ChatManager(retriever=retriever)


    question = "What does Red-Tapism mean in the context of government bureaucracy?"
    chat_history = []  # No prior conversation for this test
    result = chat_manager._condense(question, chat_history)
    print(f"Condensed question: {result}")

    result = chat_manager._retrieve(question, chat_history)
    print(f"Retrieved context: {result}")

