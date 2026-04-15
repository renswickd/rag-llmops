import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { ChatArea } from './ChatArea';
import { cn } from '@/lib/utils';

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen flex-col">
      <Header />

      <div className="relative flex min-h-0 flex-1">
        {/* Mobile hamburger — only shown below md breakpoint */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute left-2 top-2 z-30 md:hidden"
          onClick={() => setSidebarOpen((prev) => !prev)}
          aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          {sidebarOpen ? (
            <X className="h-5 w-5" />
          ) : (
            <Menu className="h-5 w-5" />
          )}
        </Button>

        {/* Sidebar — always visible on md+, toggled on mobile */}
        <div
          className={cn(
            'absolute inset-y-0 left-0 z-20 md:relative md:flex',
            sidebarOpen ? 'flex' : 'hidden'
          )}
        >
          <Sidebar />
        </div>

        {/* Mobile overlay — dims chat area when sidebar is open */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-10 bg-black/30 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <ChatArea />
      </div>
    </div>
  );
}