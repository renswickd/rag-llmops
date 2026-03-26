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
