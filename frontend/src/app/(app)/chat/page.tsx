'use client';

import { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';

import { ChatMessage } from '@/components/ChatMessage';
import { useChat } from '@/lib/useChat';
import { cn } from '@/lib/utils';

const WEEK_OPTIONS = [
  { value: '', label: 'All weeks' },
  { value: 'week_1', label: 'Week 1' },
  { value: 'week_2', label: 'Week 2' },
  { value: 'week_3', label: 'Week 3' },
  { value: 'week_4', label: 'Week 4' },
  { value: 'week_5', label: 'Week 5' },
  { value: 'week_6', label: 'Week 6' },
];

export default function ChatPage() {
  const { turns, send, busy, nodeTrace, reset } = useChat();
  const [draft, setDraft] = useState('');
  const [week, setWeek] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.trim()) return;
    const q = draft;
    setDraft('');
    await send(q, { week_filter: week || null });
  };

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col px-6 py-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="font-serif text-2xl text-ink">Chat</h1>
        <div className="flex items-center gap-3">
          <select
            value={week}
            onChange={(e) => setWeek(e.target.value)}
            className="rounded border border-parchment-warm bg-parchment px-2 py-1 text-sm text-ink"
          >
            {WEEK_OPTIONS.map((w) => (
              <option key={w.value} value={w.value}>
                {w.label}
              </option>
            ))}
          </select>
          <button
            onClick={reset}
            className="text-sm text-ink-muted hover:text-accent"
          >
            New chat
          </button>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="rounded-lg border border-parchment-warm bg-parchment p-6 text-center text-ink-muted">
            <p className="font-serif text-lg text-ink">Start by asking a question.</p>
            <p className="mt-2 text-sm">
              Try: <em>“What is the ratio decidendi in Mabo v Queensland?”</em>
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
