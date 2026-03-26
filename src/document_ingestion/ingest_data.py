from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from core.logging_config import get_logger
from core.exceptions import RagAssistantException


class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.log = get_logger(__name__)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model_loader = model_loader or ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

        self.log.info("FAISS manager initialized", index_dir=str(self.index_dir), embedding_class=type(self.embeddings).__name__)

    def _exists(self) -> bool:
        return ((self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists())

    def load(self) -> FAISS:
        try:
            if not self._exists():
                raise RagAssistantException(f"FAISS index does not exist in directory: {self.index_dir}")

            self.log.info("Loading existing FAISS index", index_dir=str(self.index_dir))
            self.vs = FAISS.load_local(str(self.index_dir), embeddings=self.embeddings, allow_dangerous_deserialization=True)
        
            return self.vs

        except Exception as e:
            self.log.error("Failed to load FAISS index", error=str(e))
            raise RagAssistantException("Failed to load FAISS index", e) from e

    def create(self, docs: List[Document]) -> FAISS:
        try:
            valid_docs = self._validate_documents(docs)
            if not valid_docs:
                raise RagAssistantException("No valid documents provided to create FAISS index")

            self.log.info("Creating new FAISS index", num_docs=len(valid_docs))
            self.vs = FAISS.from_documents(valid_docs, embedding=self.embeddings)
            self.vs.save_local(str(self.index_dir))

            self.log.info("FAISS index created successfully",index_dir=str(self.index_dir),total_docs=self.vs.index.ntotal)
            return self.vs

        except Exception as e:
            self.log.error("Failed to create FAISS index", error=str(e))
            raise RagAssistantException("Failed to create FAISS index", e) from e

    def load_or_create(self, docs: Optional[List[Document]] = None) -> FAISS:
        if not docs:
            raise RagAssistantException("No existing FAISS index and no documents provided to create one")
        if self._exists():
            return self.load()

        return self.create(docs)

    def add_documents(self, docs: List[Document]) -> int:
        try:
            if self.vs is None:
                raise RagAssistantException("Vector store is not initialized. Call load(), create(), or load_or_create() first.")

            valid_docs = self._validate_documents(docs)
            if not valid_docs:
                self.log.info("No valid documents to add")
                return 0

            self.log.info("Adding documents to FAISS index", num_docs=len(valid_docs))
            self.vs.add_documents(valid_docs)
            self.vs.save_local(str(self.index_dir))

            self.log.info("FAISS index updated successfully",index_dir=str(self.index_dir),added_docs=len(valid_docs),total_docs=self.vs.index.ntotal)
            return len(valid_docs)

        except Exception as e:
            self.log.error("Failed to add documents to FAISS index", error=str(e))
            raise RagAssistantException("Failed to add documents to FAISS index", e) from e

    @staticmethod
    def _validate_documents(docs: List[Document]) -> List[Document]:
        if not isinstance(docs, list):
            raise RagAssistantException("Documents input must be a list")

        valid_docs: List[Document] = []
        for doc in docs:
            if not isinstance(doc, Document):
                raise RagAssistantException("Each item must be a LangChain Document")
            if doc.page_content and doc.page_content.strip():
                valid_docs.append(doc)

        return valid_docs