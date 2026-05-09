from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.logging_config import get_logger

log = get_logger(__name__)


# Protocol (interface) ────

@runtime_checkable
class StorageBackend(Protocol):
    """
    Structural protocol for all storage backends.
    """

    # ── Uploaded files ──────

    def save_file(self, session_id: str, filename: str, data: bytes) -> None:
        """Persist a raw file upload under the given session."""
        ...

    def read_file(self, session_id: str, filename: str) -> bytes:
        """Return the raw bytes of a previously saved file.  Raises if not found."""
        ...

    def delete_session_files(self, session_id: str) -> None:
        """Delete all files archived under a session.  No-op if session has no files."""
        ...

    def list_session_files(self, session_id: str) -> list[str]:
        """Return filenames (not full paths) archived under a session."""
        ...

    #  Conversation history ──────

    def save_history(self, session_id: str, lines: list[str]) -> None:
        """Overwrite the JSONL history for a session with the given lines.

        Each item in `lines` is a raw JSON string (no newline suffix needed).
        Use this for bulk writes (e.g. initial migration).  For incremental
        appends during normal chat operation, use append_history().
        """
        ...

    def append_history(self, session_id: str, line: str) -> None:
        """Append a single JSON line to a session's JSONL history file.

        Called once per message (human or AI) during chat().  On Azure Blob,
        this reads-then-rewrites the full blob because Blob Storage does not
        support byte-range appends on block blobs.  This is acceptable at the
        scale of a demo project (typical session: < 100 messages).
        """
        ...

    def read_history(self, session_id: str) -> list[str]:
        """Return all lines of a session's JSONL history as a list of raw JSON strings.

        Returns an empty list if the session has no history.
        """
        ...

    def delete_history(self, session_id: str) -> None:
        """Delete the JSONL history file for a session.  No-op if not found."""
        ...

    #  Session registry ──────

    def save_registry(self, data: str) -> None:
        """Overwrite session_registry.json with the given JSON string."""
        ...

    def read_registry(self) -> str | None:
        """Return the contents of session_registry.json, or None if it does not exist."""
        ...

# Local filesystem implementation ────────

class LocalStorageBackend:
    """
    Writes to the local filesystem 

    Directory layout:
      <data_dir>/uploads/<session_id>/<filename>
      <data_dir>/history/<session_id>.jsonl
      <data_dir>/session_registry.json
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._uploads_dir = self._data_dir / "uploads"
        self._history_dir = self._data_dir / "history"
        self._registry_path = self._data_dir / "session_registry.json"

        # Ensure base directories exist at startup
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        log.info("LocalStorageBackend initialised", data_dir=str(self._data_dir))

    # ── Uploaded files ──────

    def save_file(self, session_id: str, filename: str, data: bytes) -> None:
        dest = self._uploads_dir / session_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        log.info("File saved (local)", session_id=session_id, filename=filename)

    def read_file(self, session_id: str, filename: str) -> bytes:
        path = self._uploads_dir / session_id / filename
        return path.read_bytes()

    def delete_session_files(self, session_id: str) -> None:
        import shutil
        session_dir = self._uploads_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            log.info("Session files deleted (local)", session_id=session_id)

    def list_session_files(self, session_id: str) -> list[str]:
        session_dir = self._uploads_dir / session_id
        if not session_dir.exists():
            return []
        return [p.name for p in session_dir.iterdir() if p.is_file()]

    # ── Conversation history 

    def save_history(self, session_id: str, lines: list[str]) -> None:
        path = self._history_dir / f"{session_id}.jsonl"
        path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def append_history(self, session_id: str, line: str) -> None:
        path = self._history_dir / f"{session_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_history(self, session_id: str) -> list[str]:
        path = self._history_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
        return [l for l in lines if l]

    def delete_history(self, session_id: str) -> None:
        path = self._history_dir / f"{session_id}.jsonl"
        if path.exists():
            path.unlink()
            log.info("History deleted (local)", session_id=session_id)

    # ── Session registry ────

    def save_registry(self, data: str) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(data, encoding="utf-8")

    def read_registry(self) -> str | None:
        if not self._registry_path.exists():
            return None
        return self._registry_path.read_text(encoding="utf-8")
