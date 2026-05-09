import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.logging_config import get_logger

log = get_logger(__name__)

_REGISTRY_FILENAME = "session_registry.json"


class SessionRegistry:
    """
    Thread-safe, file-backed session registry.

    Schema (one entry per session):
    {
        "<session_id>": {
            "session_id": str,
            "created_at": str,          # ISO-8601 UTC
            "documents": [
                {
                    "file_name": str,
                    "chunks_created": int
                }
            ]
        }
    }
    """

    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / _REGISTRY_FILENAME
        self._lock = threading.Lock()
        self._ensure_file()
        log.info("SessionRegistry initialised", path=str(self._path))

    def _ensure_file(self) -> None:
        """Create the registry file with an empty dict if it does not exist."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register(self, session_id: str, file_name: str, chunks_created: int) -> None:
        """
        Add a document to a session entry, creating the session if it does not exist.
        Safe to call multiple times for the same session (additional uploads).
        """
        with self._lock:
            data = self._read()
            if session_id not in data:
                data[session_id] = {
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "documents": [],
                }
            data[session_id]["documents"].append({
                "file_name": file_name,
                "chunks_created": chunks_created,
            })
            self._write(data)
            log.info("Session registered", session_id=session_id, file_name=file_name, chunks=chunks_created)

    def list_sessions(self) -> list[dict]:
        """Return all session metadata, ordered newest-first."""
        with self._lock:
            data = self._read()
        return sorted(data.values(), key=lambda s: s["created_at"], reverse=True)

    def get(self, session_id: str) -> Optional[dict]:
        """Return metadata for a single session, or None if not found."""
        with self._lock:
            return self._read().get(session_id)

    def delete(self, session_id: str) -> bool:
        """Remove a session from the registry. Returns True if it existed."""
        with self._lock:
            data = self._read()
            if session_id not in data:
                return False
            del data[session_id]
            self._write(data)
            log.info("Session deleted from registry", session_id=session_id)
            return True

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._read()


if __name__ == "__main__":
    # Phase 5 Step 7 smoke test — runs after SessionRegistry is migrated to
    # accept storage= instead of data_dir=.  Until then this block shows the
    # target state.  Run with:  uv run python core/session_store.py
    import tempfile
    from pathlib import Path
    from core.storage import LocalStorageBackend

    print("=== Step 7 SessionRegistry storage smoke ===")

    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalStorageBackend(data_dir=Path(tmp))
        registry = SessionRegistry(storage=backend)

        # 1. Registry is empty on first use
        assert registry.list_sessions() == []
        assert registry.read_registry() is None or registry.list_sessions() == []
        print("  [1] fresh registry starts empty")

        # 2. register() creates a session entry and persists it
        registry.register("sess-1", "report.pdf", chunks_created=10)
        raw = backend.read_registry()
        assert raw is not None and "sess-1" in raw
        print("  [2] register() persists session to storage")

        # 3. list_sessions() returns the entry
        sessions = registry.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-1"
        assert sessions[0]["documents"][0]["file_name"] == "report.pdf"
        print(f"  [3] list_sessions() returns {len(sessions)} session")

        # 4. second upload to same session appends a document
        registry.register("sess-1", "notes.txt", chunks_created=5)
        session = registry.get("sess-1")
        assert len(session["documents"]) == 2
        print(f"  [4] second register() appended document ({len(session['documents'])} total)")

        # 5. get() returns a single session by ID
        result = registry.get("sess-1")
        assert result is not None and result["session_id"] == "sess-1"
        assert registry.get("nonexistent") is None
        print("  [5] get() returns session or None correctly")

        # 6. exists() reflects registry state
        assert registry.exists("sess-1") is True
        assert registry.exists("ghost") is False
        print("  [6] exists() returns True/False correctly")

        # 7. registry survives a restart — data is in storage
        registry2 = SessionRegistry(storage=backend)
        assert registry2.exists("sess-1"), "data must survive registry restart via storage"
        print("  [7] data survives SessionRegistry restart (reads from storage)")

        # 8. delete() removes from registry and persists the removal
        deleted = registry.delete("sess-1")
        assert deleted is True
        assert registry.exists("sess-1") is False
        assert "sess-1" not in backend.read_registry()
        print("  [8] delete() removes session and persists change to storage")

        # 9. delete() returns False for unknown session
        assert registry.delete("ghost") is False
        print("  [9] delete() returns False for nonexistent session")

    print("=== Step 7 SessionRegistry storage smoke PASSED ===")