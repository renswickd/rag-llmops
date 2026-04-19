from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from core.exceptions import RagAssistantException
from conversation.chat_manager import ChatManager


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"source": "rag.txt"}
        )
    ]
    return retriever


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_chat_success(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
):
    mock_load_llm.return_value = MagicMock()

    mock_answer_chain = MagicMock()
    mock_answer_chain.invoke.return_value = "This is the final answer."
    mock_build_answer_chain.return_value = mock_answer_chain

    mock_condense_chain = MagicMock()
    mock_condense_chain.invoke.return_value = "What is RAG?"
    mock_build_condense_chain.return_value = mock_condense_chain

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        condense_question=True,
    )

    result = manager.chat("What is RAG?", session_id="test-session")

    assert result["answer"] == "This is the final answer."
    assert result["session_id"] == "test-session"
    assert result["history_len"] == 2
    assert result["standalone_q"] == "What is RAG?"
    assert result["sources"] == [{"source": "rag.txt"}]

    # Phase A: session_id is now forwarded to the retriever for per-session filtering
    mock_retriever.retrieve.assert_called_once_with(
        "What is RAG?",
        top_k=None,
        session_id="test-session",
    )


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_chat_empty_question_raises(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
):
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
    )

    with pytest.raises(RagAssistantException, match="Question must not be empty"):
        manager.chat("", session_id="test-session")


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_condense_returns_raw_question_when_no_history(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
):
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        condense_question=True,
    )

    result = manager._condense("Explain FAISS", chat_history=[])

    assert result == "Explain FAISS"


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_persist_turn_writes_jsonl_file(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
    tmp_path,
):
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    history_dir = tmp_path / "history"
    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        history_dir=history_dir,
    )

    manager._persist_turn("test-session", "Hello", "Hi there")

    jsonl_path = history_dir / "test-session.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2

    import json
    assert json.loads(lines[0])["role"] == "human"
    assert json.loads(lines[1])["role"] == "ai"


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_get_history_reads_jsonl_file(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
    tmp_path,
):
    import json
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    jsonl_path = history_dir / "test-session.jsonl"
    jsonl_path.write_text(
        json.dumps({"role": "human", "content": "Q?", "timestamp": "2026-01-01T00:00:00+00:00"}) + "\n"
        + json.dumps({"role": "ai",    "content": "A.", "timestamp": "2026-01-01T00:00:01+00:00"}) + "\n"
    )

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        history_dir=history_dir,
    )

    messages = manager.get_history("test-session")

    assert len(messages) == 2
    assert messages[0]["role"] == "human"
    assert messages[1]["role"] == "ai"


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_get_history_returns_empty_list_when_no_file(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
    tmp_path,
):
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        history_dir=tmp_path / "history",
    )

    assert manager.get_history("nonexistent-session") == []


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_load_persisted_history_restores_sessions(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
    tmp_path,
):
    import json
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "session-abc.jsonl").write_text(
        json.dumps({"role": "human", "content": "Hi", "timestamp": "2026-01-01T00:00:00+00:00"}) + "\n"
        + json.dumps({"role": "ai",  "content": "Hello", "timestamp": "2026-01-01T00:00:01+00:00"}) + "\n"
    )

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        history_dir=history_dir,
    )
    manager._load_persisted_history()

    assert "session-abc" in manager.list_sessions()
    assert len(manager._sessions["session-abc"].messages) == 2


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_clear_session_removes_existing_session(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
):
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
    )

    manager._get_or_create_history_session("test-session")

    assert "test-session" in manager.list_sessions()

    cleared = manager.clear_session("test-session")

    assert cleared is True
    assert "test-session" not in manager.list_sessions()


@patch("conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("conversation.chat_manager.ModelLoader.load_llm")
@patch.object(ChatManager, "_build_answer_chain")
@patch.object(ChatManager, "_build_condense_chain")
def test_clear_session_deletes_jsonl_file(
    mock_build_condense_chain,
    mock_build_answer_chain,
    mock_load_llm,
    mock_generate_session_id,
    mock_retriever,
    tmp_path,
):
    import json
    mock_load_llm.return_value = MagicMock()
    mock_build_answer_chain.return_value = MagicMock()
    mock_build_condense_chain.return_value = MagicMock()

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    jsonl_path = history_dir / "test-session.jsonl"
    jsonl_path.write_text(
        json.dumps({"role": "human", "content": "Hi", "timestamp": "2026-01-01T00:00:00+00:00"}) + "\n"
    )

    manager = ChatManager(
        retriever=mock_retriever,
        session_id="test-session",
        history_dir=history_dir,
    )
    manager._get_or_create_history_session("test-session")

    manager.clear_session("test-session")

    assert not jsonl_path.exists()