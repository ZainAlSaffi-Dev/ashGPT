'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from '@clerk/nextjs';

import { AuthNotReadyError, withAuth } from './api';
import { streamChat } from './streaming';
import type {
  ChatHistoryOverflow,
  Intent,
  RetrievalScope,
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
  scope?: RetrievalScope | null;
  /** Called the first time the backend returns a session_id — typically used
   *  by the ``/chat`` (no-id) landing route to ``router.replace`` to
   *  ``/chat/<id>`` so the URL becomes the source of truth.  */
  onSessionCreated?: (sessionId: string, turns: ChatTurn[]) => void;
  /** Called whenever a streamed assistant turn commits server-side. */
  onTurnCommitted?: (sessionId: string, turns: ChatTurn[]) => void;
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
  const turnsRef = useRef<ChatTurn[]>(opts.initialTurns ?? []);
  const busyRef = useRef(false);
  const sessionIdRef = useRef<string | null>(opts.initialSessionId ?? null);
  const abortRef = useRef<AbortController | null>(null);
  const sendSeqRef = useRef(0);
  const mountedRef = useRef(false);
  const callbacksRef = useRef({
    onSessionCreated: opts.onSessionCreated,
    onTurnCommitted: opts.onTurnCommitted,
  });

  useEffect(() => {
    callbacksRef.current = {
      onSessionCreated: opts.onSessionCreated,
      onTurnCommitted: opts.onTurnCommitted,
    };
  }, [opts.onSessionCreated, opts.onTurnCommitted]);

  const replaceTurns = useCallback(
    (nextOrUpdater: ChatTurn[] | ((prev: ChatTurn[]) => ChatTurn[])) => {
      const next =
        typeof nextOrUpdater === 'function'
          ? nextOrUpdater(turnsRef.current)
          : nextOrUpdater;
      turnsRef.current = next;
      if (mountedRef.current) setTurns(next);
    },
    [],
  );

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  // If the parent route switches sessionId (sidebar nav, browser back),
  // rebind. ``initialTurns`` is server-driven so we also reset turns when the
  // session reference changes to avoid leaking previous-session messages.
  const lastInitialSessionId = useRef<string | null | undefined>(opts.initialSessionId);
  const lastInitialTurns = useRef<ChatTurn[] | undefined>(opts.initialTurns);
  useEffect(() => {
    if (opts.initialSessionId !== lastInitialSessionId.current) {
      abortRef.current?.abort();
      abortRef.current = null;
      busyRef.current = false;
      lastInitialSessionId.current = opts.initialSessionId;
      sessionIdRef.current = opts.initialSessionId ?? null;
      setSessionId(opts.initialSessionId ?? null);
      replaceTurns(opts.initialTurns ?? []);
      setNodeTrace([]);
      setError(null);
      setBusy(false);
      lastInitialTurns.current = opts.initialTurns;
      return;
    }
    if (
      opts.initialTurns &&
      opts.initialTurns !== lastInitialTurns.current &&
      !busyRef.current
    ) {
      replaceTurns(opts.initialTurns);
      setError(null);
    }
    lastInitialTurns.current = opts.initialTurns;
  }, [opts.initialSessionId, opts.initialTurns, replaceTurns]);

  const send = useCallback(
    async (query: string, sendOpts?: { week_filter?: string | null }) => {
      if (!query.trim() || busyRef.current) return;
      const seq = sendSeqRef.current + 1;
      sendSeqRef.current = seq;
      busyRef.current = true;
      setBusy(true);
      setError(null);
      setNodeTrace([]);
      const controller = new AbortController();
      abortRef.current = controller;

      const userId = crypto.randomUUID();
      const assistantId = crypto.randomUUID();
      replaceTurns((t) => [
        ...t,
        { id: userId, role: 'user', content: query },
        { id: assistantId, role: 'assistant', content: '', streaming: true },
      ]);

      const updateAssistant = (patch: Partial<ChatTurn>) =>
        replaceTurns((t) =>
          t.map((x) => (x.id === assistantId ? { ...x, ...patch } : x)),
        );

      const isCurrent = () =>
        mountedRef.current &&
        sendSeqRef.current === seq &&
        !controller.signal.aborted;

      try {
        await withAuth(getToken, (token) =>
          streamChat(
            {
              query,
              session_id: sessionIdRef.current,
              week_filter: sendOpts?.week_filter ?? null,
              scope: opts.scope ?? null,
            },
            {
              onNode: (n) => {
                if (isCurrent()) setNodeTrace((trace) => [...trace, n]);
              },
              onSources: (sources) => {
                if (isCurrent()) updateAssistant({ sources });
              },
              onIRAC: (irac) => {
                if (isCurrent()) updateAssistant({ irac });
              },
              onMermaid: (mermaid) => {
                if (isCurrent()) updateAssistant({ mermaid });
              },
              onVerification: (verification) => {
                if (isCurrent()) updateAssistant({ verification });
              },
              onHistoryOverflow: (overflow) =>
                isCurrent() && updateAssistant({ historyOverflow: overflow }),
              onAnswerChunk: (text) =>
                isCurrent() &&
                replaceTurns((t) =>
                  t.map((x) =>
                    x.id === assistantId ? { ...x, content: x.content + text } : x,
                  ),
                ),
              onDone: ({ session_id, intent, latency_ms, final_answer }) => {
                if (!isCurrent()) return;
                const wasNewSession = !sessionIdRef.current && !!session_id;
                const committedTurns = turnsRef.current.map((x) =>
                  x.id === assistantId
                    ? {
                        ...x,
                        intent,
                        latency_ms,
                        streaming: false,
                        content: final_answer || '',
                      }
                    : x,
                );
                turnsRef.current = committedTurns;
                replaceTurns(committedTurns);
                sessionIdRef.current = session_id;
                setSessionId(session_id);
                callbacksRef.current.onTurnCommitted?.(session_id, committedTurns);
                if (wasNewSession) {
                  // First reply on the landing route — let the caller move
                  // the URL bar before the next turn so refresh / share works.
                  callbacksRef.current.onSessionCreated?.(session_id, committedTurns);
                }
              },
              onError: (detail) => {
                if (!isCurrent()) return;
                setError(detail);
                updateAssistant({ streaming: false, content: `_error: ${detail}_` });
              },
            },
            {
              token,
              signal: controller.signal,
              getFreshToken: () => getToken({ skipCache: true }),
            },
          ),
        );
      } catch (e) {
        if (controller.signal.aborted || sendSeqRef.current !== seq || !mountedRef.current) {
          return;
        }
        const msg =
          e instanceof AuthNotReadyError
            ? 'Still signing you in. Please try again in a moment.'
            : e instanceof Error
              ? e.message
              : String(e);
        setError(msg);
        updateAssistant({ streaming: false, content: `_error: ${msg}_` });
      } finally {
        if (sendSeqRef.current === seq) {
          busyRef.current = false;
          abortRef.current = null;
          if (mountedRef.current) setBusy(false);
        }
      }
    },
    [getToken, opts.scope, replaceTurns],
  );

  return { turns, send, busy, error, sessionId, nodeTrace };
}
