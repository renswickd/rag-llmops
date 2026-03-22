from pathlib import Path
import yaml
import pytest

from src.core.config import load_config


def test_load_config_from_explicit_path(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("logging:\n  level: INFO\n", encoding="utf-8")

    config = load_config(str(config_file))

    assert config["logging"]["level"] == "INFO"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("missing_config.yaml")