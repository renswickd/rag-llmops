from utils.model_loader import ModelLoader


if __name__ == "__main__":
    loader = ModelLoader()

    # llm = loader.load_llm()
    # result = llm.invoke("Who won the cricket world cup in 2019?")
    # print(f"LLM Result: {result.content}")

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {len(result)}")
