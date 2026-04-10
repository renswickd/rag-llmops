"""
Unit tests for POST /documents/upload

Medium-level coverage: HTTP contract, FAISS branch selection (create vs add),
file archival, session-ID handling, and error propagation.

The FastAPI app is built inline per-test so the real lifespan / init_services
is never triggered.  All heavy I/O (DataIngestion, FAISS) is mocked.
"""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from core.exceptions import RagAssistantException

# ─────────────────────────────────────────────
# Shared stubs
# ─────────────────────────────────────────────

FAKE_SESSION_ID = "upload_20260101_120000_abcd1234"
FAKE_DOCS   = [Document(page_content="raw document text", metadata={})]
FAKE_CHUNKS = [Document(page_content=f"chunk {i}", metadata={}) for i in range(5)]

MOCK_CONFIG = {
    "data": {
        "data_dir": "data",                          # overridden per-test via tmp_path
        "allowed_extensions": [".pdf", ".txt", ".md"],
    },
    "data_ingestion": {"chunk_size": 1000, "chunk_overlap": 150},
}


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_router_deps(tmp_path):
    """
    Replace every module-level singleton in documents.py before each test.
    tmp_path is injected so archive assertions can use the real file system.
    """
    cfg = {**MOCK_CONFIG, "data": {**MOCK_CONFIG["data"], "data_dir": str(tmp_path)}}
    with (
        patch("api.routers.documents.config", cfg),
        patch("api.routers.documents.ALLOWED_EXTENSIONS", {".pdf", ".txt", ".md"}),
        patch("api.routers.documents.generate_session_id", return_value=FAKE_SESSION_ID),
        patch("api.routers.documents.log", MagicMock()),
    ):
        yield


@pytest.fixture
def mock_faiss():
    m = MagicMock()
    m._exists.return_value = False
    m.vs = None
    m.create.return_value = MagicMock()
    m.add_documents.return_value = len(FAKE_CHUNKS)
    return m


@pytest.fixture
def client(mock_faiss):
    """Minimal FastAPI app with only the documents router; FAISS dep overridden."""
    from api.routers.documents import router
    from api.dependencies import get_faiss_manager

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_faiss_manager] = lambda: mock_faiss
    return TestClient(app)


@pytest.fixture
def stub_pipeline():
    """
    Stub DataIngestion so no real file parsing or splitting happens.
    Returns the mock DataIngestion instance for call assertions.
    """
    mock_di = MagicMock()
    mock_di.load_file.return_value = FAKE_DOCS
    mock_di.chunk_documents.return_value = FAKE_CHUNKS

    with patch("api.routers.documents.DataIngestion", return_value=mock_di):
        yield mock_di


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def upload(client, filename="report.pdf", content=b"fake pdf bytes", session_id=None):
    data = {"session_id": session_id} if session_id else {}
    return client.post(
        "/documents/upload",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        data=data,
    )


# ─────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────

def test_first_upload_creates_new_faiss_index(client, mock_faiss, stub_pipeline):
    mock_faiss._exists.return_value = False
    mock_faiss.vs = None

    resp = upload(client)

    assert resp.status_code == 200
    mock_faiss.create.assert_called_once()
    mock_faiss.add_documents.assert_not_called()


def test_subsequent_upload_adds_to_existing_index(client, mock_faiss, stub_pipeline):
    mock_faiss._exists.return_value = True
    mock_faiss.vs = MagicMock()  # already in memory

    resp = upload(client)

    assert resp.status_code == 200
    mock_faiss.add_documents.assert_called_once()
    mock_faiss.create.assert_not_called()


def test_response_body_contains_all_expected_fields(client, mock_faiss, stub_pipeline):
    mock_faiss._exists.return_value = False

    resp = upload(client, filename="notes.txt")
    body = resp.json()

    assert resp.status_code == 200
    assert body["session_id"] == FAKE_SESSION_ID
    assert body["file_name"] == "notes.txt"
    assert body["chunks_created"] == len(FAKE_CHUNKS)
    assert "notes.txt" in body["message"]
    assert "ingested" in body["message"]


def test_upload_archives_file_to_session_directory(client, mock_faiss, stub_pipeline, tmp_path):
    content = b"important document content"

    resp = upload(client, filename="report.pdf", content=content)

    assert resp.status_code == 200
    archived = tmp_path / FAKE_SESSION_ID / "report.pdf"
    assert archived.exists(), "File must be persisted to the session directory"
    assert archived.read_bytes() == content


def test_load_file_called_with_archive_path(client, mock_faiss, stub_pipeline, tmp_path):
    upload(client, filename="doc.pdf")

    expected_path = tmp_path / FAKE_SESSION_ID / "doc.pdf"
    stub_pipeline.load_file.assert_called_once_with(expected_path)


def test_chunk_documents_called_with_raw_docs(client, mock_faiss, stub_pipeline):
    upload(client)

    stub_pipeline.chunk_documents.assert_called_once_with(FAKE_DOCS)


# ─────────────────────────────────────────────
# Session-ID handling
# ─────────────────────────────────────────────

def test_uses_generated_session_id_when_none_provided(client, mock_faiss, stub_pipeline):
    resp = upload(client, session_id=None)

    assert resp.json()["session_id"] == FAKE_SESSION_ID


def test_preserves_caller_provided_session_id(client, mock_faiss, stub_pipeline):
    resp = upload(client, session_id="my-existing-session")

    assert resp.json()["session_id"] == "my-existing-session"


def test_generate_session_id_not_called_when_session_id_provided(client, mock_faiss, stub_pipeline):
    with patch("api.routers.documents.generate_session_id") as mock_gen:
        upload(client, session_id="explicit-session")

        mock_gen.assert_not_called()


# ─────────────────────────────────────────────
# Validation – 400 / 503
# ─────────────────────────────────────────────

@pytest.mark.parametrize("filename", ["malware.exe", "data.csv", "archive.zip", "image.png"])
def test_rejects_disallowed_file_extensions(client, filename):
    resp = upload(client, filename=filename)

    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_returns_503_when_faiss_manager_is_none():
    from api.routers.documents import router
    from api.dependencies import get_faiss_manager

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_faiss_manager] = lambda: None

    resp = TestClient(app).post(
        "/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"bytes"), "application/octet-stream")},
    )

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


# ─────────────────────────────────────────────
# Error propagation – 500
# ─────────────────────────────────────────────

def test_returns_500_when_load_file_raises(client, mock_faiss, stub_pipeline):
    stub_pipeline.load_file.side_effect = RagAssistantException("Corrupt or unreadable file")

    resp = upload(client, filename="corrupt.pdf")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Corrupt or unreadable file"


def test_returns_500_when_faiss_create_raises(client, mock_faiss, stub_pipeline):
    mock_faiss._exists.return_value = False
    mock_faiss.vs = None
    mock_faiss.create.side_effect = RagAssistantException("FAISS write failed")

    resp = upload(client)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "FAISS write failed"


def test_returns_500_when_faiss_add_documents_raises(client, mock_faiss, stub_pipeline):
    mock_faiss._exists.return_value = True
    mock_faiss.vs = MagicMock()
    mock_faiss.add_documents.side_effect = RagAssistantException("Index corrupted")

    resp = upload(client)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Index corrupted"
