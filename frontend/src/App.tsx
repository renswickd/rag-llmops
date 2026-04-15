import { useEffect } from 'react';
import { useAppStore } from '@/store/appStore';
import { Layout } from '@/components/Layout';

export default function App() {
  const theme = useAppStore((s) => s.theme);
  const refreshSessions = useAppStore((s) => s.refreshSessions);

  // Apply persisted theme to <html> on first load
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  // Load sessions from backend on startup
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  return <Layout />;
}