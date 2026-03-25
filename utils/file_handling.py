import os
import uuid
from datetime import datetime

def generate_session_id(prefix: str = "session") -> str:
    """
    Generate a unique session ID using UUID4.
    """
    os.makedirs("data/sessions", exist_ok=True)
    return f"sessions/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"