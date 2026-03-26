from utils.model_loader import ModelLoader


if __name__ == "__main__":
    # ------------------------------
    # Test LLM and Embeddings loading
    # ------------------------------
    # loader = ModelLoader()

    # llm = loader.load_llm()
    # result = llm.invoke("Who won the cricket world cup in 2019?")
    # print(f"LLM Result: {result.content}")

    # Test Embedding
    # embeddings = loader.load_embeddings()
    # print(f"Embedding Model Loaded: {embeddings}")
    # result = embeddings.embed_query("Hello, how are you?")
    # print(f"Embedding Result: {len(result)}")
    
    # -----------------------------
    # Test DocHandler
    # -----------------------------
    
    from src.document_ingestion.load_data import DocHandler
    handler = DocHandler()
    print(f"DocHandler session path: {handler.session_path}")

    saved_path = handler.archive_pdf("data/data_analysis/sample-doc-for-rag.pdf")
    print(f"PDF saved at: {saved_path}")
    text = handler.read_pdf(saved_path)
    print(f"Extracted text length: {len(text)}")