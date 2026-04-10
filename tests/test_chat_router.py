"""
Unit tests for the chat router:
  POST   /chat
  GET    /chat/sessions
  DELETE /chat/sessions/{session_id}

Medium-level coverage: HTTP contract, argument forwarding to ChatManager,
service-guard checks, Pydantic validation, and error propagation per endpoint.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.exceptions import RagAssistantException

# ─────────────────────────────────────────────
# Shared stubs
# ─────────────────────────────────────────────

FAKE_SESSION_ID = "sess-abc123"

FAKE_CHAT_RESULT = {
    "answer": "The capital of France is Paris.",
    "sources": [{"source": "geography.pdf", "page": 1}],
    "session_id": FAKE_SESSION_ID,
    "history_len": 1,
    "standalone_q": "What is the capital of France?",
}


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def silence_router_log():
    with patch("api.routers.chat.log", MagicMock()):
        yield


@pytest.fixture
def mock_chat_mgr():
    m = MagicMock()
    # Simulate a loaded FAISS index — documents have been ingested.
    m.retriever.faiss_manager.vs = MagicMock()
    m.chat.return_value = FAKE_CHAT_RESULT
    m.list_sessions.return_value = [FAKE_SESSION_ID, "sess-xyz"]
    m.clear_session.return_value = True
    return m


@pytest.fixture
def client(mock_chat_mgr):
    from api.routers.chat import router
    from api.dependencies import get_chat_manager

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_manager] = lambda: mock_chat_mgr
    return TestClient(app)


@pytest.fixture
def no_service_client():
    """Client where get_chat_manager returns None — simulates uninitialised service."""
    from api.routers.chat import router
    from api.dependencies import get_chat_manager

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_manager] = lambda: None
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

def test_list_sessions_returns_all_active_sessions(client, mock_chat_mgr):
    resp = client.get("/chat/sessions")

    assert resp.status_code == 200
    assert resp.json()["sessions"] == [FAKE_SESSION_ID, "sess-xyz"]


def test_list_sessions_returns_empty_list_when_none_active(client, mock_chat_mgr):
    mock_chat_mgr.list_sessions.return_value = []

    resp = client.get("/chat/sessions")

    assert resp.status_code == 200
    assert resp.json()["sessions"] == []


def test_list_sessions_returns_503_when_service_not_initialised(no_service_client):
    resp = no_service_client.get("/chat/sessions")

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


def test_list_sessions_returns_500_on_rag_exception(client, mock_chat_mgr):
    mock_chat_mgr.list_sessions.side_effect = RagAssistantException("Session store failure")

    resp = client.get("/chat/sessions")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Session store failure"


# ─────────────────────────────────────────────
# DELETE /chat/sessions/{session_id}
# ─────────────────────────────────────────────

def test_delete_session_returns_200_with_confirmation(client):
    resp = client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == FAKE_SESSION_ID
    assert FAKE_SESSION_ID in body["message"]


def test_delete_session_calls_clear_session_with_correct_id(client, mock_chat_mgr):
    client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    mock_chat_mgr.clear_session.assert_called_once_with(FAKE_SESSION_ID)


def test_delete_session_returns_404_when_session_not_found(client, mock_chat_mgr):
    mock_chat_mgr.clear_session.return_value = False

    resp = client.delete("/chat/sessions/nonexistent-session")

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_delete_session_returns_503_when_service_not_initialised(no_service_client):
    resp = no_service_client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    assert resp.status_code == 503
    assert "not initialised" in resp.json()["detail"]


def test_delete_session_returns_404_on_rag_exception(client, mock_chat_mgr):
    # The router maps RagAssistantException to 404 for this endpoint (not 500).
    mock_chat_mgr.clear_session.side_effect = RagAssistantException("Clear operation failed")

    resp = client.delete(f"/chat/sessions/{FAKE_SESSION_ID}")

    assert resp.status_code == 404
