from fastapi import APIRouter
from core.config import load_config

router = APIRouter(tags=["health"])
config = load_config()


@router.get("/health")
def health_check():
    """Returns the current health status of the API."""
    # DUMMY COMMENT - to test docker compose rebuild
    return {
        "status": "ok",
        "app": config["app"]["name"],
        "version": config["app"]["version"],
        "environment": config["app"]["environment"],
    }
