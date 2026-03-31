import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader

from core.config import load_config
from core.logging_config import get_logger
from utils.file_handling import generate_session_id
from core.exceptions import RagAssistantException
from src.document_ingestion.faiss_manager import FaissManager

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
            chunk_overlap: int = config["data_ingestion"]["chunk_overlap"],
            session_id: Optional[str] = None
        ):

        self.log = get_logger(__name__)
        self.data_dir = Path(data_dir)
        self.faiss_manager = faiss_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DataIngestion initialized", session_id=self.session_id, session_path=self.session_path)

        if not self.data_dir.exists():
            raise RagAssistantException(f"Data directory does not exist: {self.data_dir}")
        self.log.info("DataIngestion initialized", data_dir=str(self.data_dir), chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    
    def load_documents(self) -> List[Document]:
        try:
            documents: List[Document] = []
            log.info("Starting document loading", data_dir=str(self.data_dir))
            log.info("Data directory contents", files=[str(f) for f in self.data_dir.rglob("*") if f.is_file()])

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
                    archive_path = self.archive_file_in_session_path(file_path)
                    log.info("File loaded and archived", file=str(file_path), archive_path=archive_path, num_docs=len(docs))

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
            raise RagAssistantException("Failed to load documents", e)

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
            raise RagAssistantException("Failed to chunk documents", e)
    
    
    def archive_file_in_session_path(self, uploaded_file) -> str:
        try:
            # Determine filename from common attributes
            if hasattr(uploaded_file, "filename") and getattr(uploaded_file, "filename"):
                filename = os.path.basename(getattr(uploaded_file, "filename"))
            elif isinstance(uploaded_file, str):
                filename = os.path.basename(uploaded_file)
            elif hasattr(uploaded_file, "name"):
                filename = os.path.basename(getattr(uploaded_file, "name"))
            else:
                filename = str(uploaded_file)

            # os.makedirs(self.session_path, exist_ok=True)
            save_path = os.path.join(self.session_path, filename)

            # Handle different uploaded_file types
            if isinstance(uploaded_file, str):
                # uploaded_file is a filesystem path
                with open(uploaded_file, "rb") as src, open(save_path, "wb") as dst:
                    dst.write(src.read())
            elif hasattr(uploaded_file, "getbuffer"):
                # io.BytesIO or similar
                buf = uploaded_file.getbuffer()
                with open(save_path, "wb") as f:
                    f.write(buf)
            elif hasattr(uploaded_file, "stream"):
                # werkzeug FileStorage or similar
                data = uploaded_file.stream.read()
                if isinstance(data, str):
                    data = data.encode()
                with open(save_path, "wb") as f:
                    f.write(data)
            elif isinstance(uploaded_file, (bytes, bytearray)):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file)
            else:
                # Fallback: write string representation
                with open(save_path, "wb") as f:
                    f.write(str(uploaded_file).encode())

            self.log.info("Archived file successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            self.log.error("Failed to archive file", error=str(e), session_id=self.session_id)
            raise RagAssistantException(f"Failed to archive file: {str(e)}", e)
        
    def ingest(self) -> int:
        try:
            raw_docs = self.load_documents()
            chunked_docs = self.chunk_documents(raw_docs)

            self.faiss_manager.load_or_create(chunked_docs)
            self.log.info("Data ingestion completed successfully", total_chunks=len(chunked_docs))
            return len(chunked_docs)

        except Exception as e:
            self.log.error("Data ingestion failed", error=str(e))
            raise RagAssistantException("Data ingestion failed", e)

if __name__ == "__main__":
    # Data Ingestion smoke test
    from src.document_ingestion.faiss_manager import FaissManager
    faiss_manager = FaissManager(index_dir=Path("faiss_smoke_index"))
    data_ingestion = DataIngestion(data_dir=Path("data/sample_docs"),
                                      faiss_manager=faiss_manager,
                                      chunk_size=500, 
                                      chunk_overlap=50)
    
    num_chunks = data_ingestion.ingest()
    print(f"Data ingestion completed with {num_chunks} chunks created and indexed.")