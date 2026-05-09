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