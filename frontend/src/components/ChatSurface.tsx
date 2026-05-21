'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useQueryClient } from '@tanstack/react-query';
import type { Route } from 'next';
import { Send } from 'lucide-react';

import { ChatMessage } from '@/components/ChatMessage';
import { OnboardingChecklist } from '@/components/OnboardingChecklist';
import { SkeletonList } from '@/components/ui/Skeleton';
import { turnsToCachedMessages, upsertCachedSession } from '@/lib/chat-cache';
import { useOnboarding } from '@/lib/queries';
import { useChat, type ChatTurn } from '@/lib/useChat';
import type { Message, SessionSummary } from '@/lib/types';
import { cn } from '@/lib/utils';

export interface ChatSurfaceProps {
  /** Pass the existing session id (route param) to continue the conversation. */
  initialSessionId?: string | null;
  /** Server-hydrated turns shown above the live in-progress assistant bubble. */
  initialTurns?: ChatTurn[];
  /** True while a known session's history is loading for the first time. */
  historyLoading?: boolean;
}

/**
 * Shared chat UI used by both the ``/chat`` landing route (new chat) and
 * ``/chat/[sessionId]`` (rehydrated conversation). The landing route does not
 * pass a session id; once the first ``done`` event lands we ``router.replace``
 * to the dynamic route so the URL is the source of truth.
 */
export function ChatSurface({
  initialSessionId,
  initialTurns,
  historyLoading = false,
}: ChatSurfaceProps) {
  const router = useRouter();
  const queryClient = useQueryClient();

  const { turns, send, busy, nodeTrace, sessionId } = useChat({
    initialSessionId,
    initialTurns,
    onTurnCommitted: (id, committedTurns) => {
      queryClient.setQueryData<Message[]>(
        ['messages', id],
        turnsToCachedMessages(committedTurns),
      );
      queryClient.setQueryData<SessionSummary[]>(['sessions'], (sessions) =>
        upsertCachedSession(sessions, id, committedTurns),
      );
    },
    onSessionCreated: (id) => {
      // First reply on the landing route — promote to the dynamic URL so
      // reloads / sharing work. Replace (not push) so the back button still
      // lands the user on whatever they came from.
      router.replace(`/chat/${id}` as Route);
    },
  });
  const onboarding = useOnboarding({
    enabled: !historyLoading && turns.length === 0,
  });

  // Once a streamed turn lands, invalidate the cached message history for
  // this session so a tab reload / browser-back fetches the persisted
  // assistant row (with sources / irac / mermaid). Local state already has
  // everything for the current view — this just primes the cache.
  const lastBusyRef = useRef(busy);
  useEffect(() => {
    if (lastBusyRef.current && !busy && sessionId) {
      void queryClient.invalidateQueries({ queryKey: ['messages', sessionId] });
      void queryClient.invalidateQueries({ queryKey: ['sessions'] });
    }
    lastBusyRef.current = busy;
  }, [busy, sessionId, queryClient]);

  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll only while the user is parked at the bottom. If they scroll
  // up mid-stream we leave them alone until they scroll back down.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!stickToBottomRef.current) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 80;
  };

  // Textarea auto-grow (rows 2 → max ~8).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 8 * 24)}px`;
  }, [draft]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.trim() || historyLoading) return;
    const q = draft;
    setDraft('');
    stickToBottomRef.current = true;
    await send(q);
  };

  return (
    <div className="mx-auto flex h-full w-full max-w-5xl flex-col px-4 py-6 lg:px-8">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="font-serif text-2xl text-ink">Chat</h1>
        <button
          onClick={() => {
            if (!busy) router.push('/chat');
          }}
          disabled={busy}
          className={cn(
            'text-sm text-ink-muted hover:text-accent',
            busy && 'cursor-not-allowed opacity-50',
          )}
        >
          New chat
        </button>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 space-y-4 overflow-y-auto pr-1"
      >
        {historyLoading ? (
          <div className="rounded-lg border border-parchment-warm bg-parchment p-4">
            <SkeletonList rows={5} />
          </div>
        ) : turns.length === 0 && (
          <>
            {!onboarding.isLoading && !onboarding.isComplete && (
              <OnboardingChecklist variant="full" />
            )}
            <div className="rounded-lg border border-parchment-warm bg-parchment p-6 text-center text-ink-muted">
              <p className="font-serif text-lg text-ink">
                {!onboarding.isLoading && onboarding.readyFilesCount === 0
                  ? 'Add a file, then ask a question.'
                  : 'Start by asking a question.'}
              </p>
              <p className="mt-2 text-sm">
                {!onboarding.isLoading && onboarding.readyFilesCount === 0
                  ? 'Chat answers are grounded in the documents in your library.'
                  : 'Ask anything grounded in your uploaded notes, cases, or statutes.'}
              </p>
            </div>
          </>
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
          ref={textareaRef}
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
          className="flex-1 resize-none overflow-y-auto rounded-lg border border-parchment-warm bg-parchment px-4 py-3 text-ink placeholder:text-ink-soft focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || historyLoading || !draft.trim()}
          className={cn(
            'flex h-12 items-center justify-center rounded-lg bg-accent px-4 text-parchment shadow transition',
            busy || historyLoading || !draft.trim()
              ? 'cursor-not-allowed opacity-50'
              : 'hover:bg-accent-hover',
          )}
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  );
}
