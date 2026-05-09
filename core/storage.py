from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.logging_config import get_logger
from core.config import load_config

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

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

# Azure Blob Storage implementation ────────────

class AzureBlobStorageBackend:
    """
    Writes to Azure Blob Storage using the `azure-storage-blob` SDK.

    Container layout:
      uploads/   -> blobs named  "<session_id>/<filename>"
      history/   -> blobs named  "<session_id>.jsonl"
      registry/  -> one blob     "session_registry.json"

    The connection string is read from the AZURE_STORAGE_CONNECTION_STRING environment variable.  
    This is already set in ACA as a secret reference (stored as ACA secret `storage-conn-str`).

    Thread safety: BlobServiceClient is thread-safe; ContainerClient instances created on demand are lightweight and cheap.
    """
    config = load_config()
    UPLOADS_CONTAINER = config["storage"]["blob_containers"]["uploads"]
    HISTORY_CONTAINER = config["storage"]["blob_containers"]["history"]
    REGISTRY_CONTAINER = config["storage"]["blob_containers"]["registry"]
    REGISTRY_BLOB_NAME = config["storage"]["blob_containers"]["registry_blob_name"]

    def __init__(self, connection_string: str) -> None:
        
        self._ResourceNotFoundError = ResourceNotFoundError
        self._service = BlobServiceClient.from_connection_string(connection_string)
        log.info("AzureBlobStorageBackend initialised")

    def _container(self, name: str):
        """Return a ContainerClient for the named container."""
        return self._service.get_container_client(name)

    def _blob(self, container: str, blob_name: str):
        """Return a BlobClient for a specific blob."""
        return self._service.get_blob_client(container=container, blob=blob_name)

    # ── Uploaded files ────────────

    def save_file(self, session_id: str, filename: str, data: bytes) -> None:
        blob_name = f"{session_id}/{filename}"
        client = self._blob(self.UPLOADS_CONTAINER, blob_name)
        client.upload_blob(data, overwrite=True)
        log.info("File saved (blob)", session_id=session_id, filename=filename, blob=blob_name)

    def read_file(self, session_id: str, filename: str) -> bytes:
        blob_name = f"{session_id}/{filename}"
        client = self._blob(self.UPLOADS_CONTAINER, blob_name)
        stream = client.download_blob()
        return stream.readall()

    def delete_session_files(self, session_id: str) -> None:
        container = self._container(self.UPLOADS_CONTAINER)
        prefix = f"{session_id}/"
        blobs = list(container.list_blobs(name_starts_with=prefix))
        for blob in blobs:
            container.delete_blob(blob.name)
            log.debug("Blob deleted", blob=blob.name)
        log.info("Session files deleted (blob)", session_id=session_id, count=len(blobs))

    def list_session_files(self, session_id: str) -> list[str]:
        container = self._container(self.UPLOADS_CONTAINER)
        prefix = f"{session_id}/"
        blobs = container.list_blobs(name_starts_with=prefix)
        # Strip the "<session_id>/" prefix to return bare filenames
        return [b.name[len(prefix):] for b in blobs]

    # ── Conversation history 

    def save_history(self, session_id: str, lines: list[str]) -> None:
        blob_name = f"{session_id}.jsonl"
        content = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
        client = self._blob(self.HISTORY_CONTAINER, blob_name)
        client.upload_blob(content, overwrite=True)

    def append_history(self, session_id: str, line: str) -> None:
        """
        Read-modify-write pattern: download existing content, append the new line, upload the result.
        """
        blob_name = f"{session_id}.jsonl"
        client = self._blob(self.HISTORY_CONTAINER, blob_name)

        # Download existing content (empty bytes if blob doesn't exist yet)
        try:
            existing = client.download_blob().readall()
        except self._ResourceNotFoundError:
            existing = b""

        new_content = existing + (line + "\n").encode("utf-8")
        client.upload_blob(new_content, overwrite=True)

    def read_history(self, session_id: str) -> list[str]:
        blob_name = f"{session_id}.jsonl"
        client = self._blob(self.HISTORY_CONTAINER, blob_name)
        try:
            content = client.download_blob().readall().decode("utf-8")
        except self._ResourceNotFoundError:
            # Blob does not exist yet — normal for a new session
            return []
        lines = [l.strip() for l in content.splitlines()]
        return [l for l in lines if l]

    def delete_history(self, session_id: str) -> None:
        blob_name = f"{session_id}.jsonl"
        client = self._blob(self.HISTORY_CONTAINER, blob_name)
        try:
            client.delete_blob()
            log.info("History deleted (blob)", session_id=session_id)
        except self._ResourceNotFoundError:
            pass  # blob doesn't exist — that's fine

    # ── Session registry ──────────

    def save_registry(self, data: str) -> None:
        client = self._blob(self.REGISTRY_CONTAINER, self.REGISTRY_BLOB_NAME)
        client.upload_blob(data.encode("utf-8"), overwrite=True)
        log.debug("Registry saved (blob)")

    def read_registry(self) -> str | None:
        client = self._blob(self.REGISTRY_CONTAINER, self.REGISTRY_BLOB_NAME)
        try:
            return client.download_blob().readall().decode("utf-8")
        except self._ResourceNotFoundError:
            # Blob does not exist yet — return None so callers create a fresh registry
            return None


# Factory ─────────

def create_storage_backend(backend: str, data_dir: Path) -> StorageBackend:
    """
    Factory function called by api/dependencies.py at startup.

    Args:
        backend:  "local" or "azure_blob"
        data_dir: Path used only for LocalStorageBackend
    """
    if backend == "azure_blob":
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. "
                "Set it in .env (local testing) or as an ACA secret (production)."
            )
        return AzureBlobStorageBackend(connection_string=conn_str)
    else:
        return LocalStorageBackend(data_dir=data_dir)
