import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader

from core.config import load_config
from core.logging_config import get_logger
from core.exceptions import RagAssistantException
from document_ingestion.faiss_manager import FaissManager

log = get_logger(__name__)
load_dotenv()

config = load_config(os.getenv("CONFIG_PATH"))
log.info("YAML config loaded - in load_data.py", config_keys=list(config.keys()))

class DataIngestion:
    def __init__(
            self,
            data_dir: str | Path,
            faiss_manager: FaissManager, 
            chunk_size: int = config["data_ingestion"]["chunk_size"], 
            chunk_overlap: int = config["data_ingestion"]["chunk_overlap"]
        ):

        self.log = get_logger(__name__)
        self.data_dir = Path(data_dir)
        self.faiss_manager = faiss_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not self.data_dir.exists():
            raise RagAssistantException(f"Data directory does not exist: {self.data_dir}")
        self.log.info("DataIngestion initialized", data_dir=str(self.data_dir), chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    
    def load_documents(self) -> List[Document]:
        try:
            documents: List[Document] = []

            for file_path in self.data_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                suffix = file_path.suffix.lower()

                try:
                    if suffix == ".txt":
                        loader = TextLoader(str(file_path), encoding="utf-8")
                    elif suffix == ".md":
                        loader = UnstructuredMarkdownLoader(str(file_path))
                    elif suffix == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    else:
                        self.log.info("Skipping unsupported file", file=str(file_path), suffix=suffix)
                        continue

                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = str(file_path)
                        doc.metadata["file_name"] = file_path.name

                    documents.extend(docs)

                except Exception as e:
                    self.log.error("Failed to load file", file=str(file_path), error=str(e))

            self.log.info("Documents loaded successfully", total_docs=len(documents))
            return documents

        except Exception as e:
            self.log.error("Failed during document loading", error=str(e))
            raise RagAssistantException("Failed to load documents", e) from e

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        try:
            if not docs:
                raise RagAssistantException("No documents available for chunking")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            chunked_docs = splitter.split_documents(docs)

            self.log.info("Documents chunked successfully",original_docs=len(docs), chunked_docs=len(chunked_docs), chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,)
            return chunked_docs

        except Exception as e:
            self.log.error("Failed during document chunking", error=str(e))
            raise RagAssistantException("Failed to chunk documents", e) from e
        
    def ingest(self) -> int:
        try:
            raw_docs = self.load_documents()
            chunked_docs = self.chunk_documents(raw_docs)

            self.faiss_manager.load_or_create(chunked_docs)
            self.log.info("Data ingestion completed successfully", total_chunks=len(chunked_docs))
            return len(chunked_docs)

        except Exception as e:
            self.log.error("Data ingestion failed", error=str(e))
            raise RagAssistantException("Data ingestion failed", e) from e
