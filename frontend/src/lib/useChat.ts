'use client';

import { useCallback, useState } from 'react';

import { streamChat } from './streaming';
import type { Intent, SourceHit, VerificationReport } from './types';

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  intent?: Intent | null;
  sources?: SourceHit[];
  irac?: string;
  mermaid?: string;
  verification?: VerificationReport;
  latency_ms?: number;
  streaming?: boolean;
}

export function useChat() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [nodeTrace, setNodeTrace] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = useCallback(
    async (query: string, opts?: { week_filter?: string | null }) => {
      if (!query.trim() || busy) return;
      setBusy(true);
      setError(null);
      setNodeTrace([]);

      const userId = crypto.randomUUID();
      const assistantId = crypto.randomUUID();
      setTurns((t) => [
        ...t,
        { id: userId, role: 'user', content: query },
        { id: assistantId, role: 'assistant', content: '', streaming: true },
      ]);

      const updateAssistant = (patch: Partial<ChatTurn>) =>
        setTurns((t) => t.map((x) => (x.id === assistantId ? { ...x, ...patch } : x)));

      try {
        await streamChat(
          { query, session_id: sessionId, week_filter: opts?.week_filter ?? null },
          {
            onNode: (n) => setNodeTrace((trace) => [...trace, n]),
            onSources: (sources) => updateAssistant({ sources }),
            onIRAC: (irac) => updateAssistant({ irac }),
            onMermaid: (mermaid) => updateAssistant({ mermaid }),
            onVerification: (verification) => updateAssistant({ verification }),
            onAnswerChunk: (text) =>
              setTurns((t) =>
                t.map((x) =>
                  x.id === assistantId ? { ...x, content: x.content + text } : x,
                ),
              ),
            onDone: ({ session_id, intent, latency_ms, final_answer }) => {
              setSessionId(session_id);
              updateAssistant({
                intent,
                latency_ms,
                streaming: false,
                content: final_answer || '',
              });
            },
            onError: (detail) => {
              setError(detail);
              updateAssistant({ streaming: false, content: `_error: ${detail}_` });
            },
          },
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        updateAssistant({ streaming: false, content: `_error: ${msg}_` });
      } finally {
        setBusy(false);
      }
    },
    [busy, sessionId],
  );

  const reset = useCallback(() => {
    setTurns([]);
    setNodeTrace([]);
    setSessionId(null);
    setError(null);
  }, []);

  return { turns, send, reset, busy, error, sessionId, nodeTrace };
}
