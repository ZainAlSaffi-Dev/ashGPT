/**
 * Server-Sent Events parser for the /chat endpoint.
 *
 * Backend emits typed events: `node`, `sources`, `irac`, `mermaid`,
 * `verification`, `answer_chunk`, `done`, `error`. This module wraps the
 * `eventsource-parser` package and turns each event into a typed callback.
 */

import { createParser } from 'eventsource-parser';

import type { ChatStreamEvents, Intent, SourceHit, VerificationReport } from './types';

export interface ChatStreamHandlers {
  onNode?: (name: string) => void;
  onSources?: (sources: SourceHit[]) => void;
  onIRAC?: (irac: string) => void;
  onMermaid?: (diagram: string) => void;
  onVerification?: (report: VerificationReport) => void;
  onAnswerChunk?: (text: string) => void;
  onDone?: (payload: {
    session_id: string;
    intent: Intent | null;
    latency_ms: number;
    final_answer: string;
  }) => void;
  onError?: (detail: string) => void;
}

export async function streamChat(
  body: {
    query: string;
    session_id?: string | null;
    week_filter?: string | null;
  },
  handlers: ChatStreamHandlers,
  opts?: { token?: string; signal?: AbortSignal },
): Promise<void> {
  const headers = new Headers({
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  });
  if (opts?.token) headers.set('Authorization', `Bearer ${opts.token}`);
  const devUser =
    typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_DEV_USER : undefined;
  if (devUser && !opts?.token) headers.set('X-Dev-User', devUser);

  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? '/api';
  const res = await fetch(`${apiBase}/chat`, {
    method: 'POST',
    body: JSON.stringify(body),
    headers,
    signal: opts?.signal,
  });
  if (!res.ok || !res.body) {
    handlers.onError?.(`HTTP ${res.status}`);
    return;
  }

  await parseSSEStream(res.body, handlers);
}

export async function parseSSEStream(
  stream: ReadableStream<Uint8Array>,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const parser = createParser({
    onEvent: (e) => dispatch(e.event ?? 'message', e.data, handlers),
  });
  const decoder = new TextDecoder();
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.feed(decoder.decode(value, { stream: true }));
  }
}

export function dispatch(
  event: string,
  rawData: string,
  handlers: ChatStreamHandlers,
): void {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    parsed = rawData;
  }
  const data = (parsed ?? {}) as Record<string, unknown>;
  switch (event as keyof ChatStreamEvents | 'message') {
    case 'node':
      handlers.onNode?.(String(data.node ?? ''));
      break;
    case 'sources':
      handlers.onSources?.((data.sources as SourceHit[]) ?? []);
      break;
    case 'irac':
      handlers.onIRAC?.(String(data.irac ?? ''));
      break;
    case 'mermaid':
      handlers.onMermaid?.(String(data.diagram ?? ''));
      break;
    case 'verification':
      handlers.onVerification?.((data.report as VerificationReport) ?? {});
      break;
    case 'answer_chunk':
      handlers.onAnswerChunk?.(String(data.text ?? ''));
      break;
    case 'done':
      handlers.onDone?.({
        session_id: String(data.session_id ?? ''),
        intent: (data.intent as Intent | null) ?? null,
        latency_ms: Number(data.latency_ms ?? 0),
        final_answer: String(data.final_answer ?? ''),
      });
      break;
    case 'error':
      handlers.onError?.(String(data.detail ?? 'unknown error'));
      break;
    default:
      // Unknown event; intentionally ignored to keep forward-compatible.
      break;
  }
}
