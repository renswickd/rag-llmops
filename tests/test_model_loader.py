import pytest
from unittest.mock import patch, MagicMock
from core.exceptions import RagAssistantException
from utils.model_loader import ModelLoader


@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.load_config")
@patch("utils.model_loader.os.getenv")
def test_model_loader_init(load_getenv_mock, load_config_mock, load_dotenv_mock):
    # Mock CONFIG_PATH lookup during __init__
    load_getenv_mock.return_value = "config/config.yaml"
    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
                "temperature": 0.2,
                "max_output_tokens": 2048,
            }
        }
    }

    loader = ModelLoader()

    load_dotenv_mock.assert_called_once()
    load_config_mock.assert_called_once_with("config/config.yaml")
    assert "llm" in loader.config


@patch("utils.model_loader.ChatGroq")
@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_llm_success_for_groq(getenv_mock, load_dotenv_mock, load_config_mock, chatgroq_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "fake-api-key",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect

    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
                "temperature": 0.3,
                "max_output_tokens": 1024,
            }
        }
    }

    fake_llm = MagicMock()
    chatgroq_mock.return_value = fake_llm

    loader = ModelLoader()
    llm = loader.load_llm()

    chatgroq_mock.assert_called_once_with(
        model="llama-3.1-8b-instant",
        api_key="fake-api-key",
        temperature=0.3,
        max_tokens=1024,
    )
    assert llm == fake_llm


@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_llm_uses_default_provider_groq(getenv_mock, load_dotenv_mock, load_config_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
            "GROQ_API_KEY": "fake-api-key",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect

    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
                "temperature": 0.2,
                "max_output_tokens": 2048,
            }
        }
    }

    with patch("utils.model_loader.ChatGroq") as chatgroq_mock:
        loader = ModelLoader()
        loader.load_llm()

        chatgroq_mock.assert_called_once()


@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_llm_raises_when_provider_key_missing(getenv_mock, load_dotenv_mock, load_config_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
            "LLM_PROVIDER": "openai",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect

    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
            }
        }
    }

    loader = ModelLoader()

    with pytest.raises(RagAssistantException, match="LLM provider 'openai' not found in config"):
        loader.load_llm()


@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_llm_raises_for_unsupported_provider(getenv_mock, load_dotenv_mock, load_config_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
            "LLM_PROVIDER": "custom_provider",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect

    load_config_mock.return_value = {
        "llm": {
            "custom_provider": {
                "provider": "unsupported_vendor",
                "model_name": "some-model",
                "temperature": 0.5,
                "max_output_tokens": 500,
            }
        }
    }

    loader = ModelLoader()

    with pytest.raises(RagAssistantException, match="Unsupported LLM provider: unsupported_vendor"):
        loader.load_llm()

@patch("utils.model_loader.HuggingFaceEmbeddings")
@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_embeddings_success(getenv_mock, load_dotenv_mock, load_config_mock, hf_embeddings_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
            "HF_TOKEN": "fake-hf-token",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect
    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
            }
        },
        "embedding_model": {
            "model_name": "google/embeddinggemma-300m",
            "device": "cpu",
            "normalize_embeddings": True,
        },
    }

    fake_embeddings = MagicMock()
    hf_embeddings_mock.return_value = fake_embeddings

    loader = ModelLoader()
    embeddings = loader.load_embeddings()

    hf_embeddings_mock.assert_called_once_with(
        model_name="google/embeddinggemma-300m",
        model_kwargs={
            "device": "cpu",
            "token": "fake-hf-token",
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )
    assert embeddings == fake_embeddings


@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_embeddings_raises_when_embedding_block_missing(getenv_mock, load_dotenv_mock, load_config_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect
    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
            }
        }
    }

    loader = ModelLoader()

    with pytest.raises(RagAssistantException, match="Missing 'embedding_model' configuration block"):
        loader.load_embeddings()


@patch("utils.model_loader.HuggingFaceEmbeddings")
@patch("utils.model_loader.load_config")
@patch("utils.model_loader.load_dotenv")
@patch("utils.model_loader.os.getenv")
def test_load_embeddings_wraps_underlying_failure(getenv_mock, load_dotenv_mock, load_config_mock, hf_embeddings_mock):
    def getenv_side_effect(key, default=None):
        values = {
            "CONFIG_PATH": "config/config.yaml",
            "HF_TOKEN": "fake-hf-token",
        }
        return values.get(key, default)

    getenv_mock.side_effect = getenv_side_effect
    load_config_mock.return_value = {
        "llm": {
            "groq": {
                "provider": "groq",
                "model_name": "llama-3.1-8b-instant",
            }
        },
        "embedding_model": {
            "model_name": "google/embeddinggemma-300m",
            "device": "cpu",
            "normalize_embeddings": True,
        },
    }

    hf_embeddings_mock.side_effect = RuntimeError("model init failed")

    loader = ModelLoader()

    with pytest.raises(RagAssistantException, match="Failed to load embedding model"):
        loader.load_embeddings()