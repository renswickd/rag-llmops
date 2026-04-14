import { Loader2 } from 'lucide-react';
import { useAppStore } from '@/store/appStore';

export function StatusIndicator() {
  const isLoading = useAppStore((s) => s.isLoading);

  if (!isLoading) return null;

  return (
    <div className="flex items-center gap-2 px-4 py-2 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>Thinking…</span>
    </div>
  );
}