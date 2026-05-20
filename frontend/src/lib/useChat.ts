'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from '@clerk/nextjs';

import { streamChat } from './streaming';
import type {
  ChatHistoryOverflow,
  Intent,
  SourceHit,
  VerificationReport,
} from './types';

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
  historyOverflow?: ChatHistoryOverflow;
}

export interface UseChatOptions {
  /** Pre-existing session to continue (route param). */
  initialSessionId?: string | null;
  /** Server-hydrated turns to render before the first new send. */
  initialTurns?: ChatTurn[];
  /** Called the first time the backend returns a session_id — typically used
   *  by the ``/chat`` (no-id) landing route to ``router.replace`` to
   *  ``/chat/<id>`` so the URL becomes the source of truth.  */
  onSessionCreated?: (sessionId: string) => void;
}

export function useChat(opts: UseChatOptions = {}) {
  const { getToken } = useAuth();
  const [turns, setTurns] = useState<ChatTurn[]>(opts.initialTurns ?? []);
  const [nodeTrace, setNodeTrace] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(
    opts.initialSessionId ?? null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // If the parent route switches sessionId (sidebar nav, browser back),
  // rebind. ``initialTurns`` is server-driven so we also reset turns when the
  // session reference changes to avoid leaking previous-session messages.
  const lastInitialSessionId = useRef<string | null | undefined>(opts.initialSessionId);
  useEffect(() => {
    if (opts.initialSessionId !== lastInitialSessionId.current) {
      lastInitialSessionId.current = opts.initialSessionId;
      setSessionId(opts.initialSessionId ?? null);
      setTurns(opts.initialTurns ?? []);
      setNodeTrace([]);
      setError(null);
    }
  }, [opts.initialSessionId, opts.initialTurns]);

  const send = useCallback(
    async (query: string, sendOpts?: { week_filter?: string | null }) => {
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
        const token = (await getToken()) ?? undefined;
        await streamChat(
          { query, session_id: sessionId, week_filter: sendOpts?.week_filter ?? null },
          {
            onNode: (n) => setNodeTrace((trace) => [...trace, n]),
            onSources: (sources) => updateAssistant({ sources }),
            onIRAC: (irac) => updateAssistant({ irac }),
            onMermaid: (mermaid) => updateAssistant({ mermaid }),
            onVerification: (verification) => updateAssistant({ verification }),
            onHistoryOverflow: (overflow) =>
              updateAssistant({ historyOverflow: overflow }),
            onAnswerChunk: (text) =>
              setTurns((t) =>
                t.map((x) =>
                  x.id === assistantId ? { ...x, content: x.content + text } : x,
                ),
              ),
            onDone: ({ session_id, intent, latency_ms, final_answer }) => {
              if (!sessionId && session_id) {
                // First reply on the landing route — let the caller move
                // the URL bar before the next turn so refresh / share works.
                opts.onSessionCreated?.(session_id);
              }
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
          { token },
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        updateAssistant({ streaming: false, content: `_error: ${msg}_` });
      } finally {
        setBusy(false);
      }
    },
    [busy, sessionId, getToken, opts],
  );

  return { turns, send, busy, error, sessionId, nodeTrace };
}
