"""
Application entrypoint.

Start the server:
    python run.py

Or directly with uvicorn (useful for production):
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import uvicorn
from core.config import load_config

if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["app"]["environment"] == "dev",
    )
