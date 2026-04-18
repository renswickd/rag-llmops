import type { ChatResponse, UploadResponse, HistoryMessage, SessionMetadata } from '@/types';

// Reads from .env.local in dev, .env.production in build
// Falls back to /api/v1 if not set (works with Vite proxy and single-container deploy)
const API_BASE = import.meta.env.VITE_API_URL ?? '/api/v1';

// Generic fetch wrapper that throws on non-OK responses
async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error ${response.status}: ${error}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () =>
    request<{ app_name: string; version: string; environment: string }>(
      `${API_BASE}/health`
    ),

  uploadDocument: (file: File, sessionId?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (sessionId) form.append('session_id', sessionId);
    return request<UploadResponse>(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: form,
    });
  },

  chat: (question: string, sessionId: string, topK?: number) =>
    request<ChatResponse>(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        session_id: sessionId,
        ...(topK !== undefined && { top_k: topK }),
      }),
    }),

  listSessions: () =>
    request<{ sessions: SessionMetadata[] }>(`${API_BASE}/chat/sessions`),

  deleteSession: (sessionId: string) =>
    request<{ message: string }>(`${API_BASE}/chat/sessions/${sessionId}`, {
      method: 'DELETE',
    }),
  
  getHistory: (sessionId: string) =>
    request<{ session_id: string; messages: HistoryMessage[] }>(
      `${API_BASE}/chat/sessions/${sessionId}/history`
    ),
};