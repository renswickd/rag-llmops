import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store/appStore';
import { MessageBubble } from './MessageBubble';
import { ScrollArea } from '@/components/ui/scroll-area';

export function MessageList() {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const messages = useAppStore((s) => s.messages);
  const isHydrating = useAppStore((s) => s.isHydrating);
  const bottomRef = useRef<HTMLDivElement>(null);

  const currentMessages = activeSessionId
    ? (messages[activeSessionId] ?? [])
    : [];

  // Auto-scroll to newest message whenever the list grows
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentMessages.length]);

  return (
    <ScrollArea className="flex-1">
      <div className="flex flex-col gap-4 p-4">
        {!activeSessionId ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Upload a document to start chatting
          </div>
        ) : isHydrating ? (
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-sm text-muted-foreground">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
            <p>Loading conversation history…</p>
          </div>
        ) : currentMessages.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
            <p>No messages yet.</p>
            <p>Ask a question about your uploaded document below.</p>
          </div>
        ) : (
          currentMessages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        {/* Invisible anchor — auto-scroll targets this */}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}