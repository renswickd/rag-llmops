"""
Unit tests for api/dependencies.py — init_services() and dependency providers.

Coverage:
  - Local storage backend is created when APP_ENVIRONMENT != production
  - Azure storage backend is forced when APP_ENVIRONMENT=production
  - YAML backend value is used in non-production environments
  - Storage singleton is created before ChatManager
  - get_storage() returns the created singleton
  - Missing AZURE_STORAGE_CONNECTION_STRING raises RuntimeError at startup
  - ChatManager receives storage= (not history_dir=)
  - SessionRegistry receives storage=
  - _load_persisted_history is called with known_session_ids from registry
"""
from unittest.mock import MagicMock, call, patch

import pytest

import api.dependencies as deps


def _fake_faiss(exists: bool = False) -> MagicMock:
    m = MagicMock()
    m._exists.return_value = exists
    return m


@pytest.fixture(autouse=True)
def reset_singletons(tmp_path):
    """Reset module-level singletons and patch config before each test."""
    deps._model_loader = None
    deps._faiss_manager = None
    deps._retriever = None
    deps._chat_manager = None
    deps._session_registry = None
    deps._storage = None

    cfg = {
        "api": {"index_dir": str(tmp_path / "faiss")},
        "data": {"data_dir": str(tmp_path / "data")},
        "storage": {"backend": "local"},
    }
    with patch.object(deps, "config", cfg), patch.object(deps, "log", MagicMock()):
        yield


def _run_init(monkeypatch, *, app_env="dev", session_ids=None,
              storage=None, faiss_exists=False):
    """Helper: patch all heavy constructors, call init_services(), return mocks."""
    if session_ids is None:
        session_ids = []

    mock_storage = storage or MagicMock(name="storage")
    mock_chat_manager = MagicMock(name="chat_manager")
    mock_session_registry = MagicMock(name="session_registry")
    mock_session_registry.list_sessions.return_value = [
        {"session_id": sid} for sid in session_ids
    ]

    monkeypatch.setenv("APP_ENVIRONMENT", app_env)

    with (
        patch.object(deps, "create_storage_backend", return_value=mock_storage) as mock_create,
        patch.object(deps, "ModelLoader", return_value=MagicMock()),
        patch.object(deps, "FaissManager", return_value=_fake_faiss(faiss_exists)),
        patch.object(deps, "Retriever", return_value=MagicMock()),
        patch.object(deps, "ChatManager", return_value=mock_chat_manager),
        patch.object(deps, "SessionRegistry", return_value=mock_session_registry),
    ):
        deps.init_services()
        return mock_create, mock_chat_manager, mock_session_registry, mock_storage


# ─────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────

def test_local_backend_selected_in_dev(monkeypatch):
    mock_create, *_ = _run_init(monkeypatch, app_env="dev")
    assert mock_create.call_args.kwargs["backend"] == "local"


def test_azure_backend_forced_in_production(monkeypatch):
    mock_create, *_ = _run_init(monkeypatch, app_env="production")
    assert mock_create.call_args.kwargs["backend"] == "azure_blob"


def test_yaml_backend_used_in_non_production(monkeypatch):
    """Non-production always reads backend from config, never hard-codes azure_blob."""
    mock_create, *_ = _run_init(monkeypatch, app_env="staging")
    assert mock_create.call_args.kwargs["backend"] == "local"


# ─────────────────────────────────────────────
# Singleton creation and ordering
# ─────────────────────────────────────────────

def test_get_storage_returns_created_singleton(monkeypatch):
    mock_storage = MagicMock(name="storage")
    _run_init(monkeypatch, storage=mock_storage)
    assert deps.get_storage() is mock_storage


def test_storage_is_created_before_chat_manager(monkeypatch):
    events = []
    monkeypatch.setenv("APP_ENVIRONMENT", "dev")

    def record_storage(**kwargs):
        events.append("storage")
        return MagicMock(name="storage")

    def record_chat_manager(**kwargs):
        events.append("chat_manager")
        return MagicMock()

    with (
        patch.object(deps, "create_storage_backend", side_effect=record_storage),
        patch.object(deps, "ModelLoader", return_value=MagicMock()),
        patch.object(deps, "FaissManager", return_value=_fake_faiss()),
        patch.object(deps, "Retriever", return_value=MagicMock()),
        patch.object(deps, "ChatManager", side_effect=record_chat_manager),
        patch.object(deps, "SessionRegistry", return_value=MagicMock()),
    ):
        deps.init_services()

    assert events.index("storage") < events.index("chat_manager")


# ─────────────────────────────────────────────
# Constructor arguments — final wiring
# ─────────────────────────────────────────────

def test_chat_manager_receives_storage_not_history_dir(monkeypatch):
    mock_storage = MagicMock(name="storage")
    monkeypatch.setenv("APP_ENVIRONMENT", "dev")

    with (
        patch.object(deps, "create_storage_backend", return_value=mock_storage),
        patch.object(deps, "ModelLoader", return_value=MagicMock()),
        patch.object(deps, "FaissManager", return_value=_fake_faiss()),
        patch.object(deps, "Retriever", return_value=MagicMock()),
        patch.object(deps, "ChatManager", return_value=MagicMock()) as mock_cm_cls,
        patch.object(deps, "SessionRegistry", return_value=MagicMock()),
    ):
        deps.init_services()

    call_kwargs = mock_cm_cls.call_args.kwargs
    assert "storage" in call_kwargs
    assert call_kwargs["storage"] is mock_storage
    assert "history_dir" not in call_kwargs


def test_session_registry_receives_storage(monkeypatch):
    mock_storage = MagicMock(name="storage")

    with (
        patch.object(deps, "create_storage_backend", return_value=mock_storage),
        patch.object(deps, "ModelLoader", return_value=MagicMock()),
        patch.object(deps, "FaissManager", return_value=_fake_faiss()),
        patch.object(deps, "Retriever", return_value=MagicMock()),
        patch.object(deps, "ChatManager", return_value=MagicMock()),
        patch.object(deps, "SessionRegistry", return_value=MagicMock()) as mock_reg_cls,
    ):
        deps.init_services()

    call_kwargs = mock_reg_cls.call_args.kwargs
    assert "storage" in call_kwargs
    assert call_kwargs["storage"] is mock_storage
    assert "data_dir" not in call_kwargs


def test_load_persisted_history_called_with_registry_session_ids(monkeypatch):
    """History is hydrated using the session IDs from the registry, not from disk."""
    _, mock_chat_manager, *_ = _run_init(
        monkeypatch, session_ids=["sess-a", "sess-b"]
    )
    mock_chat_manager._load_persisted_history.assert_called_once_with(
        known_session_ids=["sess-a", "sess-b"]
    )


def test_load_persisted_history_called_with_empty_list_when_no_sessions(monkeypatch):
    _, mock_chat_manager, *_ = _run_init(monkeypatch, session_ids=[])
    mock_chat_manager._load_persisted_history.assert_called_once_with(known_session_ids=[])


# ─────────────────────────────────────────────
# Startup failure propagation
# ─────────────────────────────────────────────

def test_startup_fails_when_azure_backend_has_no_connection_string(monkeypatch):
    monkeypatch.setenv("APP_ENVIRONMENT", "production")

    with patch.object(
        deps,
        "create_storage_backend",
        side_effect=RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set"),
    ):
        with pytest.raises(RuntimeError, match="AZURE_STORAGE_CONNECTION_STRING"):
            deps.init_services()
