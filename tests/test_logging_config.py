import logging

from src.core.logging_config import setup_logging, get_logger


def test_setup_logging_adds_handlers():
    config = {
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
            "file_enabled": False,
            "console_enabled": True,
            "file_name_prefix": "test"
        }
    }

    setup_logging(config)
    root_logger = logging.getLogger()

    assert len(root_logger.handlers) >= 1


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert logger is not None