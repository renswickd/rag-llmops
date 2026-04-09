from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    file_name: str
    chunks_created: int
    message: str
