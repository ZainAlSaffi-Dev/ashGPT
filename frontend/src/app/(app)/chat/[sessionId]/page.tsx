'use client';

import { useQuery } from '@tanstack/react-query';
import { useAuth } from '@clerk/nextjs';
import { use } from 'react';

import { ChatSurface } from '@/components/ChatSurface';
import { listMessages } from '@/lib/api';
import type { ChatTurn } from '@/lib/useChat';

export const runtime = 'edge';

interface PageProps {
  params: Promise<{ sessionId: string }>;
}

export default function ChatSessionPage({ params }: PageProps) {
  const { sessionId } = use(params);
  const { getToken } = useAuth();

  const messagesQuery = useQuery({
    queryKey: ['messages', sessionId],
    queryFn: async () => {
      const token = (await getToken()) ?? undefined;
      return listMessages(sessionId, token);
    },
  });

  if (messagesQuery.isLoading) {
    return (
      <div className="mx-auto flex h-full max-w-4xl items-center justify-center px-6 py-6 text-ink-muted">
        Loading conversation…
      </div>
    );
  }
  if (messagesQuery.isError) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-6 text-sm text-red-700">
        Couldn’t load this conversation. Try refreshing or starting a new chat.
      </div>
    );
  }

  const turns: ChatTurn[] = (messagesQuery.data ?? []).map((m) => ({
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

  return <ChatSurface initialSessionId={sessionId} initialTurns={turns} />;
}
