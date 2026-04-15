import { useAppStore } from '@/store/appStore';
import { ThemeToggle } from './ThemeToggle';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export function Header() {
  const sessions = useAppStore((s) => s.sessions);
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const switchSession = useAppStore((s) => s.switchSession);

  return (
    <header className="flex shrink-0 items-center justify-between border-b bg-background px-6 py-3">
      <div className="flex items-center gap-2">
        <span className="text-lg font-bold">RAG Assistant</span>
      </div>

      <div className="flex items-center gap-3">
        {sessions.length > 0 && (
          <Select
            value={activeSessionId ?? ''}
            onValueChange={switchSession}
          >
            <SelectTrigger className="w-36 text-xs font-mono">
              <SelectValue placeholder="Select session" />
            </SelectTrigger>
            <SelectContent>
              {sessions.map((s) => (
                <SelectItem key={s} value={s} className="font-mono text-xs">
                  {s.slice(0, 12)}…
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}