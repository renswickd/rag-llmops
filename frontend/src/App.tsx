import { useEffect } from 'react';
import { useAppStore } from '@/store/appStore';
import { Button } from '@/components/ui/button';

export default function App() {
  const { theme, setTheme, refreshSessions, sessions, error } = useAppStore();

  // Initialise theme from store on first load
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  // Load sessions from backend on startup
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  return (
    <div className="min-h-screen bg-background text-foreground p-8">
      <h1 className="text-3xl font-bold mb-4">RAG Assistant</h1>
      <p className="text-muted-foreground mb-6">Frontend scaffold — Phase 3 builds the full UI.</p>

      <div className="flex gap-4 mb-6">
        <Button onClick={() => setTheme('light')} variant={theme === 'light' ? 'default' : 'outline'}>
          Light
        </Button>
        <Button onClick={() => setTheme('dark')} variant={theme === 'dark' ? 'default' : 'outline'}>
          Dark
        </Button>
      </div>

      {error && (
        <div className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded p-4 mb-4">
          {error}
        </div>
      )}

      <div>
        <h2 className="text-xl font-semibold mb-2">Active Sessions</h2>
        {sessions.length === 0 ? (
          <p className="text-muted-foreground">No sessions yet. Upload a document via the API to create one.</p>
        ) : (
          <ul className="list-disc pl-5">
            {sessions.map((s) => <li key={s} className="font-mono text-sm">{s}</li>)}
          </ul>
        )}
      </div>
    </div>
  );
}