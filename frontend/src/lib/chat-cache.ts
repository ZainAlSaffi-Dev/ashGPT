import type { ChatTurn } from './useChat';
import type { Message, SessionSummary } from './types';

export function turnsToCachedMessages(
  turns: ChatTurn[],
  nowIso: string = new Date().toISOString(),
): Message[] {
  return turns.map((turn) => ({
    id: turn.id,
    role: turn.role,
    content: turn.content,
    intent: turn.intent ?? null,
    retrieved_chunk_ids: null,
    sources: turn.sources ?? null,
    irac: turn.irac ?? null,
    mermaid: turn.mermaid ?? null,
    verification: turn.verification ?? null,
    latency_ms: turn.latency_ms ?? null,
    created_at: nowIso,
  }));
}

export function upsertCachedSession(
  sessions: SessionSummary[] | undefined,
  sessionId: string,
  turns: ChatTurn[],
  nowIso: string = new Date().toISOString(),
): SessionSummary[] {
  const firstUserText =
    turns.find((turn) => turn.role === 'user')?.content.trim() || 'New chat';
  const title =
    firstUserText.length > 64 ? `${firstUserText.slice(0, 61)}...` : firstUserText;
  const existing = sessions ?? [];
  const next: SessionSummary = {
    id: sessionId,
    title,
    created_at: existing.find((s) => s.id === sessionId)?.created_at ?? nowIso,
    updated_at: nowIso,
  };
  return [next, ...existing.filter((session) => session.id !== sessionId)];
}
