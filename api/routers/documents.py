import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional

from api.dependencies import get_faiss_manager
from api.schemas.document import UploadResponse
from src.document_ingestion.faiss_manager import FaissManager
from src.document_ingestion.data_ingestion import DataIngestion
from core.config import load_config
from core.exceptions import RagAssistantException
from core.logging_config import get_logger
from utils.file_handling import generate_session_id

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])
config = load_config()

# ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
ALLOWED_EXTENSIONS = set(config["data"]["allowed_extensions"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or MD file to ingest"),
    session_id: Optional[str] = Form(None, description="Reuse an existing session or leave blank to create a new one"),
    faiss_mgr: FaissManager = Depends(get_faiss_manager),
):
    """
    Upload a document and ingest it into the vector store.

    What happens here:
    1. The file is saved to a temporary directory.
    2. DataIngestion loads and splits it into chunks.
    3. The chunks are embedded and added to (or used to create) the FAISS index.
    4. You get back a `session_id` to use in your `/chat` calls.

    Supported file types: PDF, TXT, Markdown.
    """
    if faiss_mgr is None:
        raise HTTPException(status_code=503, detail="Document service not initialised.")

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    sid = session_id or generate_session_id("upload")

    # Save uploaded file to a temporary directory so DataIngestion can read it.
    # We use a temp dir so we don't pollute the main data directory with raw uploads.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        contents = await file.read()
        tmp_path.write_bytes(contents)

        try:
            data_ingestion = DataIngestion(
                data_dir=Path(tmp_dir),
                faiss_manager=faiss_mgr,
                session_id=sid,
            )

            raw_docs = data_ingestion.load_documents()
            chunks = data_ingestion.chunk_documents(raw_docs)

            # Add to existing index or create a new one if this is the first upload.
            if faiss_mgr._exists() and faiss_mgr.vs is not None:
                added = faiss_mgr.add_documents(chunks)
            else:
                faiss_mgr.create(chunks)
                added = len(chunks)

            log.info("Document ingested", file=file.filename, chunks=added, session_id=sid)

        except RagAssistantException as e:
            log.error("Ingestion failed", file=file.filename, error=str(e))
            raise HTTPException(status_code=500, detail=str(e.error_message))

    return UploadResponse(
        session_id=sid,
        file_name=file.filename,
        chunks_created=added,
        message=f"'{file.filename}' ingested successfully with {added} chunks.",
    )
