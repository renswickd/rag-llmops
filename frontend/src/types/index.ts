// Mirrors api/schemas/chat.py — keep in sync with Pydantic models

export interface ChatRequest {
    question: string;
    session_id: string;
    top_k?: number;
  }
  
  export interface SourceDocument {
    page_content: string;
    metadata: Record<string, unknown>;
  }
  
  export interface ChatResponse {
    answer: string;
    sources: SourceDocument[];
    session_id: string;
    history_len: number;
    standalone_q: string;
  }
  
  // Mirrors api/schemas/document.py
  export interface UploadResponse {
    session_id: string;
    file_name: string;
    chunks_created: number;
    message: string;
  }
  
  // Frontend-only types (not in backend)
  export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: SourceDocument[];
    standalone_q?: string;
    timestamp: Date;
  }
  
  export interface UploadedFile {
    session_id: string;
    file_name: string;
    chunks_created: number;
    uploaded_at: Date;
  }
  
  export type Theme = 'light' | 'dark';