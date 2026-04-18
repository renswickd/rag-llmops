"""
Unit tests for core.session_store.SessionRegistry

Coverage:
  - File creation on first use
  - register() — new session and subsequent uploads to same session
  - list_sessions() — ordering (newest-first) and empty case
  - get() — found and not found
  - delete() — removes entry, returns True/False
  - exists() — True/False
  - Persistence — data survives a second registry instance pointed at the same file
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Suppress structlog noise in tests
@pytest.fixture(autouse=True)
def silence_log():
    with patch("core.session_store.log", MagicMock()):
        yield


@pytest.fixture
def registry(tmp_path):
    from core.session_store import SessionRegistry
    return SessionRegistry(data_dir=tmp_path)


@pytest.fixture
def registry_path(tmp_path) -> Path:
    return tmp_path / "session_registry.json"


# ─────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────

def test_creates_registry_file_on_init(tmp_path, registry_path):
    from core.session_store import SessionRegistry
    assert not registry_path.exists()

    SessionRegistry(data_dir=tmp_path)

    assert registry_path.exists()
    assert json.loads(registry_path.read_text()) == {}


def test_does_not_overwrite_existing_registry(tmp_path, registry_path):
    """A second init on the same directory must not reset an existing registry."""
    from core.session_store import SessionRegistry

    existing_data = {"sess-1": {"session_id": "sess-1", "created_at": "2026-01-01T00:00:00+00:00", "documents": []}}
    registry_path.write_text(json.dumps(existing_data))

    reg = SessionRegistry(data_dir=tmp_path)

    assert reg.exists("sess-1")


# ─────────────────────────────────────────────
# register()
# ─────────────────────────────────────────────

def test_register_creates_new_session_entry(registry):
    registry.register("sess-1", "report.pdf", chunks_created=10)

    entry = registry.get("sess-1")
    assert entry is not None
    assert entry["session_id"] == "sess-1"
    assert entry["documents"] == [{"file_name": "report.pdf", "chunks_created": 10}]


def test_register_records_creation_timestamp(registry):
    registry.register("sess-ts", "doc.txt", chunks_created=5)

    entry = registry.get("sess-ts")
    assert "created_at" in entry
    assert entry["created_at"].endswith("+00:00")  # UTC ISO-8601


def test_register_appends_document_to_existing_session(registry):
    """Uploading a second file to the same session appends, does not overwrite."""
    registry.register("sess-1", "first.pdf", chunks_created=8)
    registry.register("sess-1", "second.md", chunks_created=4)

    entry = registry.get("sess-1")
    assert len(entry["documents"]) == 2
    assert entry["documents"][1] == {"file_name": "second.md", "chunks_created": 4}


def test_register_preserves_creation_timestamp_on_second_upload(registry):
    registry.register("sess-1", "first.pdf", chunks_created=8)
    ts_before = registry.get("sess-1")["created_at"]

    registry.register("sess-1", "second.pdf", chunks_created=2)
    ts_after = registry.get("sess-1")["created_at"]

    assert ts_before == ts_after


def test_register_persists_to_disk(registry, registry_path):
    registry.register("sess-persist", "file.txt", chunks_created=3)

    raw = json.loads(registry_path.read_text())
    assert "sess-persist" in raw


# ─────────────────────────────────────────────
# list_sessions()
# ─────────────────────────────────────────────

def test_list_sessions_returns_empty_list_when_no_sessions(registry):
    assert registry.list_sessions() == []


def test_list_sessions_returns_all_registered_sessions(registry):
    registry.register("sess-a", "a.pdf", 5)
    registry.register("sess-b", "b.pdf", 3)

    sessions = registry.list_sessions()
    ids = [s["session_id"] for s in sessions]
    assert "sess-a" in ids
    assert "sess-b" in ids


def test_list_sessions_ordered_newest_first(registry):
    """Entries are sorted by created_at descending — most recent first."""
    from core.session_store import SessionRegistry
    import time

    registry.register("sess-old", "old.pdf", 1)
    time.sleep(0.01)  # ensure different timestamps
    registry.register("sess-new", "new.pdf", 2)

    sessions = registry.list_sessions()
    assert sessions[0]["session_id"] == "sess-new"
    assert sessions[1]["session_id"] == "sess-old"


# ─────────────────────────────────────────────
# get()
# ─────────────────────────────────────────────

def test_get_returns_metadata_for_known_session(registry):
    registry.register("sess-get", "doc.md", 7)

    result = registry.get("sess-get")
    assert result["session_id"] == "sess-get"


def test_get_returns_none_for_unknown_session(registry):
    assert registry.get("does-not-exist") is None


# ─────────────────────────────────────────────
# delete()
# ─────────────────────────────────────────────

def test_delete_removes_session(registry):
    registry.register("sess-del", "to_delete.pdf", 4)

    deleted = registry.delete("sess-del")

    assert deleted is True
    assert registry.get("sess-del") is None


def test_delete_returns_false_for_unknown_session(registry):
    assert registry.delete("ghost-session") is False


def test_delete_persists_removal_to_disk(registry, registry_path):
    registry.register("sess-disk-del", "file.pdf", 2)
    registry.delete("sess-disk-del")

    raw = json.loads(registry_path.read_text())
    assert "sess-disk-del" not in raw


# ─────────────────────────────────────────────
# exists()
# ─────────────────────────────────────────────

def test_exists_returns_true_for_registered_session(registry):
    registry.register("sess-exists", "doc.txt", 1)
    assert registry.exists("sess-exists") is True


def test_exists_returns_false_for_unknown_session(registry):
    assert registry.exists("never-registered") is False


def test_exists_returns_false_after_delete(registry):
    registry.register("sess-gone", "file.txt", 1)
    registry.delete("sess-gone")
    assert registry.exists("sess-gone") is False


# ─────────────────────────────────────────────
# Persistence across instances
# ─────────────────────────────────────────────

def test_data_survives_registry_restart(tmp_path):
    """A second registry instance on the same directory reads the existing data."""
    from core.session_store import SessionRegistry

    reg1 = SessionRegistry(data_dir=tmp_path)
    reg1.register("sess-survive", "important.pdf", 12)

    reg2 = SessionRegistry(data_dir=tmp_path)
    assert reg2.exists("sess-survive")
    assert reg2.get("sess-survive")["documents"][0]["file_name"] == "important.pdf"
