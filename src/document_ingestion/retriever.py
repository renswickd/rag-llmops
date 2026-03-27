from typing import Optional, List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

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
            
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:

        self._require_vs()
        k = top_k or self.top_k
 
        try:
            if self.search_type == "similarity":
                results = self._similarity_search(query, k)
            elif self.search_type == "mmr":
                results = self._mmr_search(query, k)
            elif self.search_type == "similarity_score_threshold":
                results = self._similarity_search_with_threshold(query, k)
            else:
                raise RagAssistantException(f"Unknown search_type: {self.search_type}")
 
            self.log.info(
                "Retrieval completed",
                query_preview=query[:80],
                search_type=self.search_type,
                num_results=len(results),
                k=k,
            )
            return results
 
        except Exception as e:
            self.log.error("Retrieval failed", error=str(e), query_preview=query[:80])
            raise RagAssistantException("Retrieval failed", e) from e
        
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[tuple[Document, float]]:

        self._require_vs()
        k = top_k or self.top_k
 
        try:
            results = self.faiss_manager.vs.similarity_search_with_score(query, k=k)
            self.log.info(
                "Retrieval with scores completed",
                query_preview=query[:80],
                num_results=len(results),
                k=k,
            )
            return results
        except Exception as e:
            self.log.error("Scored retrieval failed", error=str(e))
            raise RagAssistantException("Scored retrieval failed", e) from e
    
    def as_langchain_retriever(self, top_k: Optional[int] = None, search_type: Optional[str] = None) -> VectorStoreRetriever:
        
        self._require_vs()
        k = top_k or self.top_k
        stype = search_type or self.search_type
 
        search_kwargs: dict = {"k": k}
 
        if stype == "mmr":
            search_kwargs["fetch_k"] = self.fetch_k
            search_kwargs["lambda_mult"] = self.lambda_mult
        elif stype == "similarity_score_threshold":
            if self.score_threshold is None:
                raise RagAssistantException(
                    "score_threshold must be set when search_type='similarity_score_threshold'."
                )
            search_kwargs["score_threshold"] = self.score_threshold
 
        lc_retriever = self.faiss_manager.vs.as_retriever(
            search_type=stype,
            search_kwargs=search_kwargs,
        )
 
        self.log.info("LangChain retriever created", search_type=stype, search_kwargs=search_kwargs)
        return lc_retriever
    
    # ----------------
    # Helper Functions
    # ----------------
    def _similarity_search(self, query: str, k: int) -> List[Document]:
        return self.faiss_manager.vs.similarity_search(query, k=k)
 
    def _mmr_search(self, query: str, k: int) -> List[Document]:
        return self.faiss_manager.vs.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
 
    def _similarity_search_with_threshold(self, query: str, k: int) -> List[Document]:
        if self.score_threshold is None:
            raise RagAssistantException(
                "score_threshold must be set when using search_type='similarity_score_threshold'."
            )
        pairs = self.faiss_manager.vs.similarity_search_with_score(query, k=k)
        # FAISS returns L2 distances (lower = more similar). Convert to a
        # relevance-like score via 1 / (1 + distance) for threshold comparison.
        filtered = [
            doc
            for doc, dist in pairs
            if (1.0 / (1.0 + dist)) >= self.score_threshold
        ]
        self.log.info(
            "Threshold filtering applied",
            candidates=len(pairs),
            passed=len(filtered),
            threshold=self.score_threshold,
        )
        return filtered


if __name__ == "__main__":
    from src.document_ingestion.faiss_manager import FaissManager

    faiss_manager = FaissManager(index_dir="faiss_test_index")
    retriever = Retriever(faiss_manager=faiss_manager)