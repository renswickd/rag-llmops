from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from core.exceptions import RagAssistantException
from src.conversation.chat_manager import ChatManager


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


@patch("src.conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("src.conversation.chat_manager.ModelLoader.load_llm")
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

    mock_retriever.retrieve.assert_called_once_with(
        "What is RAG?",
        top_k=None,
    )


@patch("src.conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("src.conversation.chat_manager.ModelLoader.load_llm")
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


@patch("src.conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("src.conversation.chat_manager.ModelLoader.load_llm")
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


@patch("src.conversation.chat_manager.generate_session_id", return_value="test-session")
@patch("src.conversation.chat_manager.ModelLoader.load_llm")
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

    manager._get_or_create_history_session()

    assert "test-session" in manager.list_sessions()

    cleared = manager.clear_session("test-session")

    assert cleared is True
    assert "test-session" not in manager.list_sessions()