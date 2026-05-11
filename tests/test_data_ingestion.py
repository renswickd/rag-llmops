from unittest.mock import MagicMock

from ingestion.data_ingestion import DataIngestion


def test_load_file_reads_markdown_with_text_loader(tmp_path):
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Title\n\nhello markdown\n", encoding="utf-8")

    ingestion = DataIngestion(data_dir=tmp_path, faiss_manager=MagicMock(), session_id="sess-md")

    docs = ingestion.load_file(md_path)

    assert len(docs) == 1
    assert "hello markdown" in docs[0].page_content
    assert docs[0].metadata["file_name"] == "notes.md"
    assert docs[0].metadata["source"] == str(md_path)
