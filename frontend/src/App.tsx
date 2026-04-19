import { useEffect } from 'react';
import { useAppStore } from '@/store/appStore';
import { Layout } from '@/components/Layout';

export default function App() {
  const theme = useAppStore((s) => s.theme);
  const refreshSessions = useAppStore((s) => s.refreshSessions);
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const switchSession = useAppStore((s) => s.switchSession);

  // Apply persisted theme to <html> on first load
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  // Load sessions from backend on startup
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  // Hydrate history for the active session on page load.
  // activeSessionId comes from localStorage but messages are not persisted,
  // so we trigger switchSession once on mount to fetch history from the backend.
  useEffect(() => {
    if (activeSessionId) {
      switchSession(activeSessionId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally run once on mount only

  return <Layout />;
}