"""
Unit tests for Retriever.

Phase A additions:
  - retrieve() now accepts session_id and passes filter={"session_id": ...} to FAISS
  - When no session_id is provided, filter=None (no restriction)
  - All three private search helpers (_similarity_search, _mmr_search,
    _similarity_search_with_threshold) accept and forward an optional filter_dict
"""
import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


# -------------------
# Fixtures
# -------------------

MOCK_CONFIG = {
    "retriever": {
        "default_top_k": 4,
        "default_search_type": "similarity",
        "default_fetch_k": 20,
        "default_lambda_mult": 0.5,
        "supported_search_types": ["similarity", "mmr", "similarity_score_threshold"],
    }
}

DOCS = [
    Document(page_content="A"),
    Document(page_content="B"),
]


@pytest.fixture(autouse=True)
def patch_deps():
    with (
        patch("core.config.load_config", return_value=MOCK_CONFIG),
        patch("core.logging_config.get_logger", return_value=MagicMock()),
        patch("core.exceptions.RagAssistantException", side_effect=Exception),
    ):
        yield


@pytest.fixture
def Retriever():
    from ingestion.retriever import Retriever
    return Retriever


@pytest.fixture
def vs():
    vs = MagicMock()
    vs.index.ntotal = 2
    vs.similarity_search.return_value = DOCS
    vs.max_marginal_relevance_search.return_value = DOCS
    vs.similarity_search_with_score.return_value = [(d, 0.1) for d in DOCS]
    vs.as_retriever.return_value = MagicMock(spec=VectorStoreRetriever)
    return vs


@pytest.fixture
def manager(vs):
    m = MagicMock()
    m.vs = vs
    return m


# ---------------
# Core behavior
# ---------------

def test_init_defaults(Retriever, manager):
    r = Retriever(manager)
    assert (r.top_k, r.search_type) == (4, "similarity")


def test_init_invalid_search_type(Retriever, manager):
    with pytest.raises(Exception):
        Retriever(manager, search_type="bad")


def test_initialize_calls_manager(Retriever, manager):
    r = Retriever(manager)
    r.initialize(DOCS)
    manager.load_or_create.assert_called_once()


def test_require_vs_raises(Retriever):
    r = Retriever(MagicMock(vs=None))
    with pytest.raises(Exception):
        r._require_vs()


# ---------------
# Retrieval — routing
# ---------------

@pytest.mark.parametrize("stype,method", [
    ("similarity", "similarity_search"),
    ("mmr", "max_marginal_relevance_search"),
])
def test_retrieve_routes_to_correct_faiss_method(Retriever, manager, vs, stype, method):
    r = Retriever(manager, search_type=stype)
    r.retrieve("q")
    getattr(vs, method).assert_called_once()


def test_retrieve_with_scores(Retriever, manager, vs):
    r = Retriever(manager)
    out = r.retrieve_with_scores("q")
    assert isinstance(out[0][0], Document)


def test_retrieve_without_vs_raises(Retriever):
    r = Retriever(MagicMock(vs=None))
    with pytest.raises(Exception):
        r.retrieve("q")


# ---------------
# Session filtering (Phase A)
# ---------------

def test_retrieve_with_session_id_passes_filter_to_faiss(Retriever, manager, vs):
    """When session_id is provided, FAISS is called with filter={"session_id": ...}."""
    r = Retriever(manager)
    r.retrieve("find this", session_id="upload_20260418_100000_abc00001")

    vs.similarity_search.assert_called_once_with(
        "find this",
        k=4,
        filter={"session_id": "upload_20260418_100000_abc00001"},
    )


def test_retrieve_without_session_id_passes_no_filter(Retriever, manager, vs):
    """When no session_id is given, FAISS is called with filter=None (unrestricted)."""
    r = Retriever(manager)
    r.retrieve("find this")

    vs.similarity_search.assert_called_once_with("find this", k=4, filter=None)


def test_retrieve_mmr_with_session_id_passes_filter(Retriever, manager, vs):
    r = Retriever(manager, search_type="mmr")
    r.retrieve("q", session_id="sess-1")

    vs.max_marginal_relevance_search.assert_called_once_with(
        "q",
        k=4,
        fetch_k=20,
        lambda_mult=0.5,
        filter={"session_id": "sess-1"},
    )


def test_retrieve_mmr_without_session_id_passes_no_filter(Retriever, manager, vs):
    r = Retriever(manager, search_type="mmr")
    r.retrieve("q")

    vs.max_marginal_relevance_search.assert_called_once_with(
        "q",
        k=4,
        fetch_k=20,
        lambda_mult=0.5,
        filter=None,
    )


def test_retrieve_respects_top_k_override(Retriever, manager, vs):
    r = Retriever(manager, top_k=4)
    r.retrieve("q", top_k=2, session_id="sess-x")

    vs.similarity_search.assert_called_once_with("q", k=2, filter={"session_id": "sess-x"})


# ---------------
# LangChain adapter
# ---------------

def test_as_langchain_retriever(Retriever, manager, vs):
    r = Retriever(manager, top_k=3)
    r.as_langchain_retriever()
    vs.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


def test_as_langchain_threshold_requires_score(Retriever, manager):
    r = Retriever(manager, search_type="similarity_score_threshold")
    with pytest.raises(Exception):
        r.as_langchain_retriever()


# ---------------
# Threshold logic
# ---------------

def test_threshold_filters_low_score_results(Retriever, manager, vs):
    vs.similarity_search_with_score.return_value = [
        (DOCS[0], 0.0),   # score = 1/(1+0) = 1.0 — passes threshold 0.6
        (DOCS[1], 10.0),  # score = 1/(1+10) ≈ 0.09 — fails threshold
    ]

    r = Retriever(
        manager,
        search_type="similarity_score_threshold",
        score_threshold=0.6,
        top_k=2,
    )

    out = r.retrieve("q")
    assert len(out) == 1


def test_threshold_with_session_id_passes_filter(Retriever, manager, vs):
    vs.similarity_search_with_score.return_value = [(DOCS[0], 0.0)]

    r = Retriever(
        manager,
        search_type="similarity_score_threshold",
        score_threshold=0.5,
    )
    r.retrieve("q", session_id="sess-filter")

    vs.similarity_search_with_score.assert_called_once_with(
        "q", k=4, filter={"session_id": "sess-filter"}
    )


# ---------------
# Delegation helpers
# ---------------

def test_private_helpers_delegate_to_faiss(Retriever, manager, vs):
    r = Retriever(manager)
    r._similarity_search("q", 1)
    r._mmr_search("q", 1)

    vs.similarity_search.assert_called()
    vs.max_marginal_relevance_search.assert_called()


def test_private_helpers_forward_filter_dict(Retriever, manager, vs):
    r = Retriever(manager)
    filt = {"session_id": "s1"}

    r._similarity_search("q", 2, filt)
    vs.similarity_search.assert_called_with("q", k=2, filter=filt)

    r._mmr_search("q", 2, filt)
    vs.max_marginal_relevance_search.assert_called_with(
        "q", k=2, fetch_k=20, lambda_mult=0.5, filter=filt
    )
