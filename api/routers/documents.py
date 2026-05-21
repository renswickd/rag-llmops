import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, Response
from api.limiter import limiter
from api.dependencies import get_faiss_manager, get_session_registry, get_storage
from api.schemas.document import UploadResponse
from ingestion.faiss_manager import FaissManager
from ingestion.data_ingestion import DataIngestion
from core.session_store import SessionRegistry
from core.storage import StorageBackend
from core.config import load_config
from core.exceptions import RagAssistantException
from core.logging_config import get_logger
from utils.file_handling import generate_session_id

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])
config = load_config()
_upload_rpm = config.get("rate_limiting", {}).get("upload_rpm", 5)

ALLOWED_EXTENSIONS = set(config["data"]["allowed_extensions"])


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(f"{_upload_rpm}/minute")
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="PDF, TXT, or MD file to ingest"),
    session_id: Optional[str] = Form(None, description="Reuse an existing session or leave blank to create a new one"),
    faiss_mgr: FaissManager = Depends(get_faiss_manager),
    session_registry: SessionRegistry = Depends(get_session_registry),
    storage: StorageBackend = Depends(get_storage),
):
    """
    Upload a document and ingest it into the vector store.

    What happens here:
    1. The file bytes are persisted to the storage backend (local or Blob).
    2. A temporary local copy is written so DataIngestion can parse it.
    3. DataIngestion.load_file() parses the temp file into LangChain documents.
    4. DataIngestion.chunk_documents() splits them into indexed chunks.
    5. The chunks are embedded and added to (or used to create) the FAISS index.
    6. You get back a `session_id` to use in your `/chat` calls.

    Supported file types: PDF, TXT, Markdown.
    """
    if faiss_mgr is None or session_registry is None:
        raise HTTPException(status_code=503, detail="Document service not initialised.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    sid = session_id or generate_session_id("upload")
    contents = await file.read()

    # # Persist the uploaded file to the configured data directory.
    # # This archive survives process restarts and is the canonical copy for re-ingestion.
    # data_dir = Path(config["data"]["data_dir"])
    # session_dir = data_dir / "uploads" / sid      # data/uploads/<session_id>/
    # session_dir.mkdir(parents=True, exist_ok=True)
    # archive_path = session_dir / file.filename
    # archive_path.write_bytes(contents)
    # log.info("File archived to session directory", file=file.filename, archive_path=str(archive_path), session_id=sid)\

    # 1. Persist to the configured storage backend (local disk or Azure Blob).
    storage.save_file(session_id=sid, filename=file.filename, data=contents)
    log.info("File archived to storage backend", file=file.filename, session_id=sid)

    # 2. Write a temporary local copy for DataIngestion (which needs a real Path).
    #    Using a temp file avoids coupling DataIngestion to the storage abstraction.
    #    The temp file is deleted in the finally block.

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        data_ingestion = DataIngestion(
            data_dir=tmp_path.parent,
            faiss_manager=faiss_mgr,
            session_id=sid,
        )

        # load_file() targets the single archived file directly — no directory scan.
        raw_docs = data_ingestion.load_file(tmp_path)
        chunks = data_ingestion.chunk_documents(raw_docs)

        # Add to existing index or create a new one if this is the first upload.
        if faiss_mgr._exists() and faiss_mgr.vs is not None:
            added = faiss_mgr.add_documents(chunks)
        else:
            faiss_mgr.create(chunks)
            added = len(chunks)

        session_registry.register(
            session_id=sid,
            file_name=file.filename,
            chunks_created=added,
        )

        log.info("Document ingested", file=file.filename, chunks=added, session_id=sid)
        
        return UploadResponse(
            session_id=sid,
            file_name=file.filename,
            chunks_created=added,
            message=f"'{file.filename}' ingested successfully with {added} chunks.",
        )

    except RagAssistantException as e:
        log.error("Ingestion failed", file=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e.error_message))
    
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()