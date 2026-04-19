import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { api } from '@/api/client';
import type { Message, UploadedFile, SessionMetadata, Theme } from '@/types';

interface AppState {
  // State
  activeSessionId: string | null;
  sessions: SessionMetadata[];
  sessionMetadata: Record<string, SessionMetadata>;  // fast lookup by session_id
  messages: Record<string, Message[]>;   // sessionId -> messages for that session
  isLoading: boolean;
  isHydrating: boolean;                  // true while fetching history on session switch
  uploadedFiles: UploadedFile[];
  theme: Theme;
  error: string | null;

  // Actions
  sendMessage: (question: string) => Promise<void>;
  uploadDocument: (file: File) => Promise<void>;
  switchSession: (sessionId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
  setTheme: (theme: Theme) => void;
  clearError: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      activeSessionId: null,
      sessions: [],
      sessionMetadata: {},
      messages: {},
      isLoading: false,
      isHydrating: false,
      uploadedFiles: [],
      theme: 'light',
      error: null,

      sendMessage: async (question: string) => {
        const { activeSessionId, messages } = get();
        if (!activeSessionId) {
          set({ error: 'No active session. Upload a document first.' });
          return;
        }

        const userMessage: Message = {
          id: crypto.randomUUID(),
          role: 'user',
          content: question,
          timestamp: new Date(),
        };

        // Append user message immediately (optimistic update)
        set({
          isLoading: true,
          error: null,
          messages: {
            ...messages,
            [activeSessionId]: [
              ...(messages[activeSessionId] ?? []),
              userMessage,
            ],
          },
        });

        try {
          const response = await api.chat(question, activeSessionId);
          const assistantMessage: Message = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: response.answer,
            sources: response.sources,
            standalone_q: response.standalone_q,
            timestamp: new Date(),
          };

          set((state) => ({
            isLoading: false,
            messages: {
              ...state.messages,
              [activeSessionId]: [
                ...(state.messages[activeSessionId] ?? []),
                assistantMessage,
              ],
            },
          }));
        } catch (err) {
          set({
            isLoading: false,
            error: err instanceof Error ? err.message : 'Failed to send message',
          });
        }
      },

      uploadDocument: async (file: File) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.uploadDocument(file);
          // Build a minimal SessionMetadata for immediate display.
          // refreshSessions() will sync the authoritative version from the backend.
          const newSession: SessionMetadata = {
            session_id: response.session_id,
            created_at: new Date().toISOString(),
            documents: [{ file_name: response.file_name, chunks_created: response.chunks_created }],
          };
          set((state) => ({
            isLoading: false,
            activeSessionId: response.session_id,
            sessions: state.sessions.some(s => s.session_id === response.session_id)
              ? state.sessions
              : [newSession, ...state.sessions],
            sessionMetadata: {
              ...state.sessionMetadata,
              [response.session_id]: state.sessionMetadata[response.session_id] ?? newSession,
            },
            uploadedFiles: [
              {
                session_id: response.session_id,
                file_name: response.file_name,
                chunks_created: response.chunks_created,
                uploaded_at: new Date(),
              },
              ...state.uploadedFiles,
            ],
            messages: {
              ...state.messages,
              [response.session_id]: state.messages[response.session_id] ?? [],
            },
          }));
        } catch (err) {
          set({
            isLoading: false,
            error: err instanceof Error ? err.message : 'Upload failed',
          });
        }
      },

      switchSession: async (sessionId: string) => {
        set({ activeSessionId: sessionId, error: null });

        // Re-hydrate from backend if no messages cached for this session
        const { messages } = get();
        if (!messages[sessionId] || messages[sessionId].length === 0) {
          set({ isHydrating: true });
          try {
            const response = await api.getHistory(sessionId);
            if (response.messages.length > 0) {
              // Convert backend HistoryMessage to frontend Message
              const hydrated: Message[] = response.messages.map((m) => ({
                id: crypto.randomUUID(),
                role: m.role === 'human' ? 'user' : 'assistant',
                content: m.content,
                timestamp: new Date(m.timestamp),
              }));
              set((state) => ({
                messages: { ...state.messages, [sessionId]: hydrated },
              }));
            }
          } catch {
            // History re-hydration is best-effort — do not block the session switch
          } finally {
            set({ isHydrating: false });
          }
        }
      },

      deleteSession: async (sessionId: string) => {
        try {
          await api.deleteSession(sessionId);
          set((state) => {
            const newSessions = state.sessions.filter(s => s.session_id !== sessionId);
            const newMessages = { ...state.messages };
            const newMetadata = { ...state.sessionMetadata };
            delete newMessages[sessionId];
            delete newMetadata[sessionId];
            return {
              sessions: newSessions,
              sessionMetadata: newMetadata,
              messages: newMessages,
              activeSessionId:
                state.activeSessionId === sessionId
                  ? (newSessions[0]?.session_id ?? null)
                  : state.activeSessionId,
            };
          });
        } catch (err) {
          set({
            error: err instanceof Error ? err.message : 'Delete failed',
          });
        }
      },

      refreshSessions: async () => {
        try {
          const response = await api.listSessions();
          const metadata: Record<string, SessionMetadata> = {};
          for (const s of response.sessions) {
            metadata[s.session_id] = s;
          }
          set({ sessions: response.sessions, sessionMetadata: metadata });
        } catch (err) {
          set({
            error: err instanceof Error ? err.message : 'Failed to load sessions',
          });
        }
      },

      setTheme: (theme: Theme) => {
        set({ theme });
        // Apply to <html> element for Tailwind's class-based dark mode
        document.documentElement.classList.toggle('dark', theme === 'dark');
        localStorage.setItem('theme', theme);
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'rag-app-storage-v2',         // bumped from v1 — sessions now SessionMetadata[]
      partialize: (state) => ({          // Only persist these fields across page reloads
        theme: state.theme,
        activeSessionId: state.activeSessionId,
        sessions: state.sessions,           // SessionMetadata[] — safe to persist
        sessionMetadata: state.sessionMetadata,
        uploadedFiles: state.uploadedFiles,
        // Do NOT persist messages — they can grow large and are session-specific
      }),
    }
  )
);