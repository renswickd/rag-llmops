from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

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
    1. The file is archived to data_dir/<session_id>/<filename> for durability.
    2. DataIngestion.load_file() parses the archived file into LangChain documents.
    3. DataIngestion.chunk_documents() splits them into indexed chunks.
    4. The chunks are embedded and added to (or used to create) the FAISS index.
    5. You get back a `session_id` to use in your `/chat` calls.

    Supported file types: PDF, TXT, Markdown.
    """
    if faiss_mgr is None:
        raise HTTPException(status_code=503, detail="Document service not initialised.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    sid = session_id or generate_session_id("upload")
    contents = await file.read()

    # Persist the uploaded file to the configured data directory.
    # This archive survives process restarts and is the canonical copy for re-ingestion.
    data_dir = Path(config["data"]["data_dir"])
    session_dir = data_dir / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    archive_path = session_dir / file.filename
    archive_path.write_bytes(contents)
    log.info("File archived to session directory", file=file.filename, archive_path=str(archive_path), session_id=sid)

    try:
        # DataIngestion is initialised with the root data_dir so its session_path
        # resolves to the already-created data_dir/sid/ — no new directories are made.
        data_ingestion = DataIngestion(
            data_dir=data_dir,
            faiss_manager=faiss_mgr,
            session_id=sid,
        )

        # load_file() targets the single archived file directly — no directory scan.
        raw_docs = data_ingestion.load_file(archive_path)
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
