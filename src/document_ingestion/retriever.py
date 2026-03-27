from typing import Optional, List
from langchain_core.documents import Document

from core.config import load_config
from core.logging_config import get_logger
from core.exceptions import RagAssistantException
from src.document_ingestion.faiss_manager import FaissManager

config = load_config()

class Retriever:
 
    def __init__(
        self,
        faiss_manager: FaissManager,
        top_k: int = config["retriever"]["default_top_k"],
        search_type: str = config["retriever"]["default_search_type"],
        score_threshold: Optional[float] = None,
        fetch_k: int = config["retriever"]["default_fetch_k"],
        lambda_mult: float = config["retriever"]["default_lambda_mult"],
    ):
        self.log = get_logger(__name__)
        
        self.SUPPORTED_SEARCH_TYPES = config["retriever"]["supported_search_types"]
 
        if search_type not in self.SUPPORTED_SEARCH_TYPES:
            raise RagAssistantException(
                f"Unsupported search_type '{search_type}'. "
                f"Choose one of: {self.SUPPORTED_SEARCH_TYPES}"
            )
 
        self.faiss_manager = faiss_manager
        self.top_k = top_k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
 
        self.log.info(
            "Retriever created",
            top_k=self.top_k,
            search_type=self.search_type,
            score_threshold=self.score_threshold,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
    
    def initialize(self, docs: Optional[List[Document]] = None) -> "Retriever":
        try:
            self.faiss_manager.load_or_create(docs)
            self.log.info("Retriever initialized via load_or_create",total_vectors=self.faiss_manager.vs.index.ntotal)
            return self
        except Exception as e:
            self.log.error("Failed to initialize Retriever", error=str(e))
            raise RagAssistantException("Failed to initialize Retriever", e) from e
    
    def _require_vs(self) -> None:
        if self.faiss_manager.vs is None:
            raise RagAssistantException(
                "Vector store is not initialized. "
                "Call Retriever.initialize() or FaissManager.load() / create() first."
            )

if __name__ == "__main__":
    from src.document_ingestion.faiss_manager import FaissManager

    faiss_manager = FaissManager(index_dir="faiss_test_index")
    retriever = Retriever(faiss_manager=faiss_manager)