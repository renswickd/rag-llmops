import uuid
from datetime import datetime


def generate_session_id(prefix: str = "session") -> str:
    """
    Generate a unique, URL-safe session ID.
    No forward slashes — safe to use directly as a FastAPI path parameter and as a filesystem directory name.

    Format: <prefix>_YYYYMMDD_HHMMSS_<8hex>
    Example: upload_20260417_143022_a3f9c1b7
    """
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"