import { useAppStore } from '@/store/appStore';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { StatusIndicator } from './StatusIndicator';

export function ChatArea() {
  const error = useAppStore((s) => s.error);
  const clearError = useAppStore((s) => s.clearError);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Error banner */}
      {error && (
        <div className="flex items-center justify-between bg-destructive/10 px-4 py-2 text-sm text-destructive">
          <span>{error}</span>
          <button
            onClick={clearError}
            className="ml-4 text-xs underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      <MessageList />
      <StatusIndicator />
      <ChatInput />
    </div>
  );
}