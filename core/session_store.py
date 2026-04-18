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