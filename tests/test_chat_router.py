"""
Unit tests for the chat router:
  POST   /chat
  GET    /chat/sessions
  DELETE /chat/sessions/{session_id}

Medium-level coverage: HTTP contract, argument forwarding to ChatManager,
service-guard checks, Pydantic validation, and error propagation per endpoint.

Phase A changes reflected here:
  - list_sessions now reads from SessionRegistry, not ChatManager._sessions
  - delete_session performs full cleanup: registry → files → FAISS rebuild → chat history
  - delete_session 404 is driven by registry.exists(), not clear_session() return value
  - All three endpoints now depend on get_session_registry
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.exceptions import RagAssistantException

# ─────────────────────────────────────────────
# Shared stubs
# ─────────────────────────────────────────────

FAKE_SESSION_ID = "upload_20260418_100000_abc00001"

FAKE_CHAT_RESULT = {
    "answer": "The capital of France is Paris.",
    "sources": [{"source": "geography.pdf", "page": 1}],
    "session_id": FAKE_SESSION_ID,
    "history_len": 1,
    "standalone_q": "What is the capital of France?",
}

FAKE_SESSION_METADATA = [
    {
        "session_id": FAKE_SESSION_ID,
        "created_at": "2026-04-18T10:00:00+00:00",
        "documents": [{"file_name": "geo.pdf", "chunks_created": 10}],
    },
    {
        "session_id": "upload_20260418_090000_abc00002",
        "created_at": "2026-04-18T09:00:00+00:00",
        "documents": [],
    },
]


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def silence_router_log():
    with patch("api.routers.chat.log", MagicMock()):
        yield


@pytest.fixture(autouse=True)
def patch_chat_config(tmp_path):
    """Route file-system operations in delete_session to a temp directory."""
    cfg = {"data": {"data_dir": str(tmp_path)}}
    with patch("api.routers.chat.config", cfg):
        yield


@pytest.fixture
def mock_chat_mgr():
    m = MagicMock()
    m.retriever.faiss_manager.vs = MagicMock()
    m.chat.return_value = FAKE_CHAT_RESULT
    m.clear_session.return_value = True
    return m


@pytest.fixture
def mock_registry():
    m = MagicMock()
    m.list_sessions.return_value = FAKE_SESSION_METADATA
    m.exists.return_value = True
    m.delete.return_value = True
    return m


@pytest.fixture
def mock_faiss_mgr():
    return MagicMock()


@pytest.fixture
def client(mock_chat_mgr, mock_registry, mock_faiss_mgr):
    from api.routers.chat import router
    from api.dependencies import get_chat_manager, get_session_registry, get_faiss_manager

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_manager] = lambda: mock_chat_mgr
    app.dependency_overrides[get_session_registry] = lambda: mock_registry
    app.dependency_overrides[get_faiss_manager] = lambda: mock_faiss_mgr
    return TestClient(app)


@pytest.fixture
def no_service_client(mock_registry, mock_faiss_mgr):
    """Client where get_chat_manager returns None — simulates uninitialised service."""
    from api.routers.chat import router
    from api.dependencies import get_chat_manager, get_session_registry, get_faiss_manager

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_manager] = lambda: None
    app.dependency_overrides[get_session_registry] = lambda: mock_registry
    app.dependency_overrides[get_faiss_manager] = lambda: mock_faiss_mgr
    return TestClient(app)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def chat_payload(question="What is the capital of France?", session_id=FAKE_SESSION_ID, top_k=None):
    payload = {"question": question, "session_id": session_id}
    if top_k is not None:
        payload["top_k"] = top_k
    return payload


# ─────────────────────────────────────────────
# POST /chat — happy path
# ─────────────────────────────────────────────

def test_chat_returns_200_on_valid_request(client):
    resp = client.post("/chat", json=chat_payload())

    assert resp.status_code == 200


def test_chat_response_contains_all_expected_fields(client):
    resp = client.post("/chat", json=chat_payload())
    body = resp.json()

    assert body["answer"] == FAKE_CHAT_RESULT["answer"]
    assert body["sources"] == FAKE_CHAT_RESULT["sources"]
    assert body["session_id"] == FAKE_SESSION_ID
    assert body["history_len"] == 1
    assert body["standalone_q"] == FAKE_CHAT_RESULT["standalone_q"]


def test_chat_forwards_question_and_session_id_to_manager(client, mock_chat_mgr):
    client.post("/chat", json=chat_payload(question="Explain RAG", session_id="my-session"))

    mock_chat_mgr.chat.assert_called_once_with(
        question="Explain RAG",
        session_id="my-session",
        top_k=None,
    )


def test_chat_forwards_top_k_when_provided(client, mock_chat_mgr):
    client.post("/chat", json=chat_payload(top_k=6))

    mock_chat_mgr.chat.assert_called_once_with(
        question="What is the capital of France?",
        session_id=FAKE_SESSION_ID,
        top_k=6,
    )


def test_chat_passes_none_top_k_when_omitted(client, mock_chat_mgr):
    client.post("/chat", json=chat_payload())

    assert mock_chat_mgr.chat.call_args.kwargs["top_k"] is None


# ─────────────────────────────────────────────
# POST /chat — guard checks and error cases
# ─────────────────────────────────────────────

def test_chat_returns_503_when_service_not_initialised(no_service_client):
    resp = no_service_client.post("/chat", json=chat_payload())

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


def test_chat_returns_400_when_no_documents_ingested(client, mock_chat_mgr):
    mock_chat_mgr.retriever.faiss_manager.vs = None

    resp = client.post("/chat", json=chat_payload())

    assert resp.status_code == 400
    assert "Upload a document" in resp.json()["detail"]


def test_chat_returns_500_on_rag_exception(client, mock_chat_mgr):
    mock_chat_mgr.chat.side_effect = RagAssistantException("LLM call failed")

    resp = client.post("/chat", json=chat_payload())

    assert resp.status_code == 500
    assert resp.json()["detail"] == "LLM call failed"


def test_chat_rejects_empty_question(client):
    # ChatRequest enforces min_length=1 — FastAPI returns 422 before the handler runs.
    resp = client.post("/chat", json={"question": "", "session_id": FAKE_SESSION_ID})

    assert resp.status_code == 422


@pytest.mark.parametrize("top_k", [0, 21, -1])
def test_chat_rejects_out_of_range_top_k(client, top_k):
    # ChatRequest enforces ge=1, le=20 — validated by Pydantic before the handler.
    resp = client.post("/chat", json=chat_payload(top_k=top_k))

    assert resp.status_code == 422


# ─────────────────────────────────────────────
# GET /chat/sessions
# ─────────────────────────────────────────────

def test_list_sessions_returns_all_registered_sessions(client):
    """Sessions are returned as full metadata objects (Phase C)."""
    resp = client.get("/chat/sessions")

    assert resp.status_code == 200
    session_ids = [s["session_id"] for s in resp.json()["sessions"]]
    assert FAKE_SESSION_ID in session_ids
    assert "upload_20260418_090000_abc00002" in session_ids


def test_list_sessions_returns_empty_list_when_registry_empty(client, mock_registry):
    mock_registry.list_sessions.return_value = []

    resp = client.get("/chat/sessions")

    assert resp.status_code == 200
    assert resp.json()["sessions"] == []


def test_list_sessions_returns_full_metadata_objects(client):
    """Response contains SessionMetadata objects with session_id, created_at, documents."""
    resp = client.get("/chat/sessions")
    sessions = resp.json()["sessions"]

    assert all("session_id" in s and "created_at" in s and "documents" in s for s in sessions)


def test_list_sessions_returns_503_when_service_not_initialised(no_service_client):
    resp = no_service_client.get("/chat/sessions")

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


def test_list_sessions_returns_500_on_exception(client, mock_registry):
    mock_registry.list_sessions.side_effect = Exception("Registry read error")

    resp = client.get("/chat/sessions")

    assert resp.status_code == 500
    assert "Registry read error" in resp.json()["detail"]


# ─────────────────────────────────────────────
# DELETE /chat/sessions/{session_id}
# ─────────────────────────────────────────────

def test_delete_session_returns_200_with_confirmation(client):
    resp = client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == FAKE_SESSION_ID
    assert FAKE_SESSION_ID in body["message"]


def test_delete_session_checks_registry_first(client, mock_registry):
    """404 is driven by registry.exists(), not ChatManager.clear_session()."""
    mock_registry.exists.return_value = False

    resp = client.delete("/chat/sessions/nonexistent-session")

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_delete_session_calls_registry_delete(client, mock_registry):
    client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    mock_registry.delete.assert_called_once_with(FAKE_SESSION_ID)


def test_delete_session_clears_faiss_index(client, mock_faiss_mgr):
    client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    mock_faiss_mgr.clear.assert_called_once()


def test_delete_session_clears_in_memory_chat_history(client, mock_chat_mgr):
    client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    mock_chat_mgr.clear_session.assert_called_once_with(FAKE_SESSION_ID)


def test_delete_session_rebuilds_faiss_when_sessions_remain(client, mock_registry, mock_faiss_mgr, tmp_path):
    """When other sessions remain and their files exist, FAISS is rebuilt."""
    remaining_id = "upload_20260418_090000_abc00002"
    mock_registry.list_sessions.return_value = [
        {"session_id": remaining_id, "created_at": "2026-04-18T09:00:00+00:00", "documents": []},
    ]

    # Create a real file so the rebuild loop processes it
    session_dir = tmp_path / "uploads" / remaining_id
    session_dir.mkdir(parents=True)
    txt_file = session_dir / "doc.txt"
    txt_file.write_text("some content")

    with patch("api.routers.chat.DataIngestion") as mock_di_cls:
        mock_di = MagicMock()
        mock_di.load_file.return_value = []
        mock_di.chunk_documents.return_value = []
        mock_di_cls.return_value = mock_di

        client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    mock_faiss_mgr.clear.assert_called_once()


def test_delete_session_does_not_rebuild_faiss_when_no_sessions_remain(
    client, mock_registry, mock_faiss_mgr
):
    """When the deleted session was the last one, FAISS is cleared but not rebuilt."""
    mock_registry.list_sessions.return_value = []

    client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    mock_faiss_mgr.clear.assert_called_once()
    mock_faiss_mgr.create.assert_not_called()


def test_delete_session_returns_503_when_service_not_initialised(no_service_client):
    resp = no_service_client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


def test_delete_session_returns_500_on_unexpected_exception(client, mock_registry):
    mock_registry.delete.side_effect = Exception("Disk write failed")

    resp = client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    assert resp.status_code == 500
    assert "Disk write failed" in resp.json()["detail"]


# ─────────────────────────────────────────────
# GET /chat/sessions/{session_id}/history  (Phase B)
# ─────────────────────────────────────────────

FAKE_HISTORY_MESSAGES = [
    {"role": "human", "content": "What is RAG?", "timestamp": "2026-04-18T10:00:00+00:00"},
    {"role": "ai",    "content": "RAG stands for Retrieval-Augmented Generation.", "timestamp": "2026-04-18T10:00:01+00:00"},
]


def test_get_history_returns_200_with_messages(client, mock_chat_mgr):
    mock_chat_mgr.get_history.return_value = FAKE_HISTORY_MESSAGES

    resp = client.get(f"/chat/sessions/{FAKE_SESSION_ID}/history")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == FAKE_SESSION_ID
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "human"
    assert body["messages"][1]["role"] == "ai"


def test_get_history_returns_empty_list_when_no_history(client, mock_chat_mgr):
    mock_chat_mgr.get_history.return_value = []

    resp = client.get(f"/chat/sessions/{FAKE_SESSION_ID}/history")

    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_get_history_returns_503_when_service_not_initialised(no_service_client):
    resp = no_service_client.get(f"/chat/sessions/{FAKE_SESSION_ID}/history")

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


def test_get_history_returns_500_on_exception(client, mock_chat_mgr):
    mock_chat_mgr.get_history.side_effect = Exception("File read error")

    resp = client.get(f"/chat/sessions/{FAKE_SESSION_ID}/history")

    assert resp.status_code == 500
    assert "File read error" in resp.json()["detail"]
