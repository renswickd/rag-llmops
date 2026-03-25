import os
import logging

import pytest

from src.document_ingestion import data_ingestion as data_ingestion


def setup_module(module):
    logging.getLogger("src").setLevel(logging.CRITICAL)


def test_init_creates_session_path(tmp_path, monkeypatch):
    monkeypatch.setattr(data_ingestion, "load_config", lambda p: {"path": {"data_dir": str(tmp_path)}})
    monkeypatch.setattr(data_ingestion, "generate_session_id", lambda prefix: "sess-123")

    handler = data_ingestion.DocHandler(data_dir=None, session_id=None)

    assert handler.session_id == "sess-123"
    assert os.path.isdir(handler.session_path)


def test_archive_pdf_from_path(tmp_path, monkeypatch):
    monkeypatch.setattr(data_ingestion, "load_config", lambda p: {"path": {"data_dir": str(tmp_path)}})
    monkeypatch.setattr(data_ingestion, "generate_session_id", lambda prefix: "sess-path")

    # create a small dummy "pdf" file
    src_pdf = tmp_path / "input.pdf"
    src_pdf.write_bytes(b"%PDF-1.4 dummy")

    handler = data_ingestion.DocHandler()
    out_path = handler.archive_pdf(str(src_pdf))

    assert os.path.exists(out_path)
    assert out_path.endswith("input.pdf")

def test_read_pdf_mocked(monkeypatch, tmp_path):
    monkeypatch.setattr(data_ingestion, "load_config", lambda p: {"path": {"data_dir": str(tmp_path)}})
    monkeypatch.setattr(data_ingestion, "generate_session_id", lambda prefix: "sess-read")

    # Fake fitz.open to return a context manager with pages
    class FakePage:
        def __init__(self, text):
            self._t = text

        def get_text(self):
            return self._t

    class FakeDoc:
        def __init__(self, pages):
            self.page_count = len(pages)
            self._pages = pages

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_page(self, idx):
            return FakePage(self._pages[idx])

    def fake_open(path):
        return FakeDoc(["page1 text", "page2 text"])

    monkeypatch.setattr(data_ingestion.fitz, "open", fake_open)

    handler = data_ingestion.DocHandler()
    # path may be arbitrary since fitz.open is mocked
    text = handler.read_pdf(str(tmp_path / "irrelevant.pdf"))

    assert "Page 1" in text
    assert "page1 text" in text
