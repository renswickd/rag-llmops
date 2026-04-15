import { cn } from '@/lib/utils';
import { MarkdownRenderer } from './MarkdownRenderer';
import { SourceCitations } from './SourceCitations';
import type { Message } from '@/types';

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[80%] rounded-xl px-4 py-3',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-foreground'
        )}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : (
          <>
            {message.standalone_q &&
              message.standalone_q !== message.content && (
                <p className="mb-2 text-xs italic text-muted-foreground">
                  Interpreted as: &ldquo;{message.standalone_q}&rdquo;
                </p>
              )}
            <MarkdownRenderer content={message.content} />
            {message.sources && message.sources.length > 0 && (
              <SourceCitations sources={message.sources} />
            )}
          </>
        )}
      </div>
    </div>
  );
}