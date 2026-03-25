import os
import sys
from dotenv import load_dotenv
from typing import Optional
from core.config import load_config
from core.logging_config import get_logger
from core.exceptions import RagAssistantException
from utils.file_handling import generate_session_id

log = get_logger(__name__)

class DocHandler:
    """
    PDF save + read (page-wise) for analysis.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        load_dotenv()
        log.info("Environment variables loaded from .env file - in data_ingestion.py")

        self.config = load_config(os.getenv("CONFIG_PATH"))
        log.info("YAML config loaded - in data_ingestion.py", config_keys=list(self.config.keys()))
        
        # self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.data_dir = data_dir
        if not self.data_dir:
            default_data_dir = os.path.join(os.getcwd(), "data", "document_analysis")
            self.data_dir = self.config["path"]["data_dir"] or default_data_dir
            
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

if __name__ == "__main__":
    # Example usage
    handler = DocHandler()
    log.info("DocHandler example instance created", session_id=handler.session_id, session_path=handler.session_path)
