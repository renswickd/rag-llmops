import logging
import os
from datetime import datetime
from pathlib import Path
import structlog
from core.config import load_config
from dotenv import load_dotenv


_LOGGING_INITIALIZED = False

load_dotenv()

def _build_log_file_path(log_dir: str, prefix: str = "app") -> str:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    return os.path.join(log_dir, f"{prefix}-{timestamp}.log")


def setup_logging(config: dict | None = None) -> None:
    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED:
        return

    if config is None:
        # _build_log_file_path("logs", "app")
        config = load_config(os.getenv("CONFIG_PATH"))

    logging_cfg = config.get("logging", {})
    log_level = logging_cfg.get("level", "INFO").upper()
    log_dir = logging_cfg.get("log_dir", "logs")
    file_enabled = logging_cfg.get("file_enabled", True)
    console_enabled = logging_cfg.get("console_enabled", True)
    file_name_prefix = logging_cfg.get("file_name_prefix", "app")

    handlers: list[logging.Handler] = []

    if console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

    if file_enabled:
        log_file_path = _build_log_file_path(log_dir, file_name_prefix)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    for handler in handlers:
        root_logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.stdlib.add_log_level,
            structlog.processors.EventRenamer(to="event"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _LOGGING_INITIALIZED:
        setup_logging()
    return structlog.get_logger(name)