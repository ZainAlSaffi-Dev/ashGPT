'use client';

import { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';

import { ChatMessage } from '@/components/ChatMessage';
import { useChat } from '@/lib/useChat';
import { cn } from '@/lib/utils';

export default function ChatPage() {
  const { turns, send, busy, nodeTrace, reset } = useChat();
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.trim()) return;
    const q = draft;
    setDraft('');
    await send(q);
  };

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col px-6 py-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="font-serif text-2xl text-ink">Chat</h1>
        <button
          onClick={reset}
          className="text-sm text-ink-muted hover:text-accent"
        >
          New chat
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="rounded-lg border border-parchment-warm bg-parchment p-6 text-center text-ink-muted">
            <p className="font-serif text-lg text-ink">Start by asking a question.</p>
            <p className="mt-2 text-sm">
              Ask anything grounded in your uploaded notes, cases, or statutes.
            </p>
          </div>
        )}
        {turns.map((t) => (
          <ChatMessage key={t.id} turn={t} />
        ))}
        {busy && nodeTrace.length > 0 && (
          <p className="ml-1 text-xs text-ink-soft">
            running: {nodeTrace.join(' → ')}
          </p>
        )}
      </div>

      <form onSubmit={submit} className="mt-4 flex items-end gap-2">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void submit(e as unknown as React.FormEvent);
            }
          }}
          placeholder="Ask about your notes…"
          rows={2}
          className="flex-1 resize-none rounded-lg border border-parchment-warm bg-parchment px-4 py-3 text-ink placeholder:text-ink-soft focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || !draft.trim()}
          className={cn(
            'flex h-12 items-center justify-center rounded-lg bg-accent px-4 text-parchment shadow transition',
            busy || !draft.trim() ? 'cursor-not-allowed opacity-50' : 'hover:bg-accent-hover',
          )}
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  );
}
