'use client';

import { use } from 'react';

import { ChatSurface } from '@/components/ChatSurface';
import { useMessages } from '@/lib/queries';
import type { ChatTurn } from '@/lib/useChat';

export const runtime = 'edge';

interface PageProps {
  params: Promise<{ sessionId: string }>;
}

export default function ChatSessionPage({ params }: PageProps) {
  const { sessionId } = use(params);

  // ``useMessages`` is auth-gated + persisted to localStorage, so revisits
  // paint from cache immediately. ``data`` is undefined only on the very
  // first uncached visit; in that case we still mount ChatSurface with no
  // turns so the composer is interactive while the messages stream in.
  const messagesQuery = useMessages(sessionId);
  const messages = messagesQuery.data;
  const historyLoading = !messages && messagesQuery.isLoading;

  if (messagesQuery.isError && !messages) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-6 text-sm text-red-700">
        Couldn’t load this conversation. Try refreshing or starting a new chat.
      </div>
    );
  }

  const turns: ChatTurn[] = (messages ?? []).map((m) => ({
    id: m.id,
    role: m.role,
    content: m.content,
    intent: (m.intent ?? null) as ChatTurn['intent'],
    sources: m.sources ?? undefined,
    irac: m.irac ?? undefined,
    mermaid: m.mermaid ?? undefined,
    verification: m.verification ?? undefined,
    latency_ms: m.latency_ms ?? undefined,
  }));

  return (
    <ChatSurface
      initialSessionId={sessionId}
      initialTurns={messages ? turns : undefined}
      historyLoading={historyLoading}
    />
  );
}
