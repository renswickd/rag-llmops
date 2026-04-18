import { useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { MessageSquare, Trash2, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { SessionMetadata } from '@/types';

function sessionLabel(meta: SessionMetadata): string {
  if (meta.documents.length === 0) return 'No documents';
  if (meta.documents.length === 1) return meta.documents[0].file_name;
  return `${meta.documents[0].file_name} +${meta.documents.length - 1}`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function SessionList() {
  const sessions = useAppStore((s) => s.sessions);
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const switchSession = useAppStore((s) => s.switchSession);
  const deleteSession = useAppStore((s) => s.deleteSession);
  const refreshSessions = useAppStore((s) => s.refreshSessions);
  const [openDialogId, setOpenDialogId] = useState<string | null>(null);

  const handleDelete = async (sessionId: string) => {
    await deleteSession(sessionId);
    setOpenDialogId(null);
    await refreshSessions();
  };

  if (sessions.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No sessions yet. Upload a document to create one.
      </p>
    );
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Sessions
        </h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5"
          onClick={refreshSessions}
          aria-label="Refresh sessions"
          title="Refresh sessions"
        >
          <RefreshCw className="h-3 w-3" />
        </Button>
      </div>

      <ul className="space-y-1">
        {sessions.map((meta) => (
          <li
            key={meta.session_id}
            className={cn(
              'group flex cursor-pointer items-center gap-1.5 rounded px-2 py-1.5 text-xs',
              meta.session_id === activeSessionId
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-muted'
            )}
            onClick={() => switchSession(meta.session_id)}
          >
            <MessageSquare className="h-3 w-3 shrink-0" />
            <span className="flex-1 truncate" title={meta.session_id}>
              {sessionLabel(meta)}
            </span>
            <span className="shrink-0 text-[10px] opacity-50">
              {formatDate(meta.created_at)}
            </span>

            {/* Delete button — only visible on hover */}
            <Dialog
              open={openDialogId === meta.session_id}
              onOpenChange={(open) => setOpenDialogId(open ? meta.session_id : null)}
            >
              <DialogTrigger asChild>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenDialogId(meta.session_id);
                  }}
                  className={cn(
                    'rounded p-0.5 transition-opacity',
                    meta.session_id === activeSessionId
                      ? 'opacity-60 hover:opacity-100'
                      : 'opacity-0 group-hover:opacity-100 hover:text-destructive'
                  )}
                  aria-label="Delete session"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </DialogTrigger>

              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Delete session?</DialogTitle>
                  <DialogDescription>
                    This will permanently delete{' '}
                    <strong>{sessionLabel(meta)}</strong> and all its chat
                    history. This cannot be undone.
                  </DialogDescription>
                </DialogHeader>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => setOpenDialogId(null)}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={() => handleDelete(meta.session_id)}
                  >
                    Delete
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </li>
        ))}
      </ul>
    </div>
  );
}
