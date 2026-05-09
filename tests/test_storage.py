from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceNotFoundError

from core.storage import (
    AzureBlobStorageBackend,
    LocalStorageBackend,
    create_storage_backend,
)


@pytest.fixture(autouse=True)
def silence_storage_log():
    with patch("core.storage.log", MagicMock()):
        yield


def test_local_storage_file_round_trip(tmp_path):
    storage = LocalStorageBackend(data_dir=tmp_path)

    storage.save_file("sess-1", "notes.txt", b"hello world")

    assert storage.read_file("sess-1", "notes.txt") == b"hello world"


def test_local_storage_lists_session_files(tmp_path):
    storage = LocalStorageBackend(data_dir=tmp_path)
    storage.save_file("sess-1", "a.txt", b"a")
    storage.save_file("sess-1", "b.txt", b"b")

    assert sorted(storage.list_session_files("sess-1")) == ["a.txt", "b.txt"]


def test_local_storage_deletes_session_files(tmp_path):
    storage = LocalStorageBackend(data_dir=tmp_path)
    storage.save_file("sess-1", "a.txt", b"a")

    storage.delete_session_files("sess-1")

    assert storage.list_session_files("sess-1") == []
    assert not (tmp_path / "uploads" / "sess-1").exists()


def test_local_storage_history_round_trip(tmp_path):
    storage = LocalStorageBackend(data_dir=tmp_path)

    storage.append_history("sess-1", '{"role":"human","content":"Hi"}')
    storage.append_history("sess-1", '{"role":"ai","content":"Hello"}')

    assert storage.read_history("sess-1") == [
        '{"role":"human","content":"Hi"}',
        '{"role":"ai","content":"Hello"}',
    ]


def test_local_storage_delete_history_is_noop_for_missing_session(tmp_path):
    storage = LocalStorageBackend(data_dir=tmp_path)

    storage.delete_history("missing-session")

    assert storage.read_history("missing-session") == []


def test_local_storage_registry_round_trip(tmp_path):
    storage = LocalStorageBackend(data_dir=tmp_path)
    payload = '{"sess-1": {"session_id": "sess-1"}}'

    storage.save_registry(payload)

    assert storage.read_registry() == payload


def test_create_storage_backend_returns_local_backend(tmp_path):
    backend = create_storage_backend("local", data_dir=tmp_path)
    assert isinstance(backend, LocalStorageBackend)


def test_create_storage_backend_raises_when_connection_string_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_STORAGE_CONNECTION_STRING"):
        create_storage_backend("azure_blob", data_dir=tmp_path)


def test_create_storage_backend_raises_for_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="Unsupported storage backend"):
        create_storage_backend("azure-blob-typo", data_dir=tmp_path)


def test_azure_backend_save_file_uploads_blob():
    with patch("core.storage.BlobServiceClient.from_connection_string") as mock_from:
        service = MagicMock()
        blob_client = MagicMock()
        service.get_blob_client.return_value = blob_client
        mock_from.return_value = service

        storage = AzureBlobStorageBackend("UseDevelopmentStorage=true")
        storage.save_file("sess-1", "report.pdf", b"pdf-bytes")

        blob_client.upload_blob.assert_called_once_with(b"pdf-bytes", overwrite=True)


def test_azure_backend_read_history_returns_empty_list_when_blob_missing():
    with patch("core.storage.BlobServiceClient.from_connection_string") as mock_from:
        service = MagicMock()
        blob_client = MagicMock()
        blob_client.download_blob.side_effect = ResourceNotFoundError("missing")
        service.get_blob_client.return_value = blob_client
        mock_from.return_value = service

        storage = AzureBlobStorageBackend("UseDevelopmentStorage=true")

        assert storage.read_history("sess-1") == []


def test_azure_backend_delete_history_swallows_missing_blob():
    with patch("core.storage.BlobServiceClient.from_connection_string") as mock_from:
        service = MagicMock()
        blob_client = MagicMock()
        blob_client.delete_blob.side_effect = ResourceNotFoundError("missing")
        service.get_blob_client.return_value = blob_client
        mock_from.return_value = service

        storage = AzureBlobStorageBackend("UseDevelopmentStorage=true")
        storage.delete_history("sess-1")


def test_azure_backend_list_session_files_strips_prefix():
    with patch("core.storage.BlobServiceClient.from_connection_string") as mock_from:
        service = MagicMock()
        container_client = MagicMock()
        container_client.list_blobs.return_value = [
            type("BlobItem", (), {"name": "sess-1/first.pdf"})(),
            type("BlobItem", (), {"name": "sess-1/second.md"})(),
        ]
        service.get_container_client.return_value = container_client
        mock_from.return_value = service

        storage = AzureBlobStorageBackend("UseDevelopmentStorage=true")

        assert storage.list_session_files("sess-1") == ["first.pdf", "second.md"]


def test_azure_backend_read_registry_returns_none_when_missing():
    with patch("core.storage.BlobServiceClient.from_connection_string") as mock_from:
        service = MagicMock()
        blob_client = MagicMock()
        blob_client.download_blob.side_effect = ResourceNotFoundError("missing")
        service.get_blob_client.return_value = blob_client
        mock_from.return_value = service

        storage = AzureBlobStorageBackend("UseDevelopmentStorage=true")

        assert storage.read_registry() is None
