import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import Optional
from core.config import load_config
from core.logging_config import get_logger
from core.exceptions import RagAssistantException
from utils.file_handling import generate_session_id

log = get_logger(__name__)

class DocHandler:
    """
    PDF archive + read (page-wise) for analysis.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        load_dotenv()
        log.info("Environment variables loaded from .env file - in load_data.py")

        self.config = load_config(os.getenv("CONFIG_PATH"))
        log.info("YAML config loaded - in load_data.py", config_keys=list(self.config.keys()))
        
        self.data_dir = data_dir
        if not self.data_dir:
            default_data_dir = os.path.join(os.getcwd(), "data", "document_analysis")
            self.data_dir = self.config["path"]["data_dir"] or default_data_dir
            
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)
        
    def archive_pdf(self, uploaded_file) -> str:
        try:
            # Determine filename from common attributes
            if hasattr(uploaded_file, "filename") and getattr(uploaded_file, "filename"):
                filename = os.path.basename(getattr(uploaded_file, "filename"))
            elif isinstance(uploaded_file, str):
                filename = os.path.basename(uploaded_file)
            elif hasattr(uploaded_file, "name"):
                filename = os.path.basename(getattr(uploaded_file, "name"))
            else:
                filename = uploaded_file

            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")

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
            else:
                raise ValueError("Unsupported uploaded_file type")

            log.info("Input files saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save input files", error=str(e), session_id=self.session_id)
            raise RagAssistantException(f"Failed to save input files: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}") 
            text = "\n".join(text_chunks)
            log.info(f"Input read successfully with total pages{len(text_chunks)}", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read input", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise RagAssistantException(f"Could not process input files: {pdf_path}", e) from e

if __name__ == "__main__":
    # Example usage
    handler = DocHandler()
    log.info("DocHandler example instance created", session_id=handler.session_id, session_path=handler.session_path)

