from pathlib import Path
import os
import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config_path(config_path: str | None = None) -> Path:
    if config_path:
        path = Path(config_path)
    elif os.getenv("CONFIG_PATH"):
        path = Path(os.getenv("CONFIG_PATH"))
    else:
        path = get_project_root() / "config" / "config.yaml"

    return path.resolve()


def load_config(config_path: str | None = None) -> dict:
    path = resolve_config_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}