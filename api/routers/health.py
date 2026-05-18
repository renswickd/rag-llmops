import os

from fastapi import APIRouter
from core.config import load_config

router = APIRouter(tags=["health"])
config = load_config()


@router.get("/health")
def health_check():
    """Returns the current health status of the API."""
    return {
        "status": "ok",
        "app": config["app"]["name"],
        "version": config["app"]["version"],
        "environment": os.getenv("APP_ENVIRONMENT", config["app"]["environment"]),
    }
