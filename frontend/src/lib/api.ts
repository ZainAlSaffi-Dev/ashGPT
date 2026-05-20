/**
 * Typed fetch wrappers for the LawGPT backend.
 *
 * In production every call uses the relative `/api/*` path (rewritten by
 * `next.config.mjs` to the backend host). In tests the base can be overridden.
 *
 * Auth: Clerk attaches the session token automatically when the request is
 * issued from a signed-in client component via `@clerk/nextjs` helpers. For
 * the v1 we send the dev-bypass header when ``NEXT_PUBLIC_DEV_USER`` is set.
 */

import type {
  ExamPayload,
  ExamResult,
  FileMeta,
  Message,
  PresignResponse,
  SessionSummary,
} from './types';

// Resolve the API base URL with a runtime fallback. Build-time env vars
// (process.env.NEXT_PUBLIC_API_BASE) haven't been reliably plumbed through
// the Pages build pipeline, so we also detect the hosted domain and route
// directly to the edge Worker when we're on it.
export function resolveApiBase(): string {
  const buildBase = process.env.NEXT_PUBLIC_API_BASE;
  if (buildBase) return buildBase;
  if (typeof window !== 'undefined' && window.location.hostname.endsWith('pages.dev')) {
    return 'https://lawgpt-edge.hypersonic3692.workers.dev';
  }
  return '/api';
}

const API_BASE = resolveApiBase();

export class ApiError extends Error {
  constructor(public status: number, public body: string) {
    super(`API ${status}: ${body.slice(0, 200)}`);
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const devUser = process.env.NEXT_PUBLIC_DEV_USER;
  if (devUser && !token) headers.set('X-Dev-User', devUser);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set('Content-Type', headers.get('Content-Type') ?? 'application/json');
  }

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    throw new ApiError(res.status, await res.text().catch(() => ''));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// Sessions --------------------------------------------------------------------

export const listSessions = (token?: string) =>
  request<SessionSummary[]>('/sessions', { method: 'GET' }, token);

export const createSession = (title: string | null, token?: string) =>
  request<SessionSummary>(
    '/sessions',
    { method: 'POST', body: JSON.stringify({ title }) },
    token,
  );

export const listMessages = (sessionId: string, token?: string) =>
  request<Message[]>(`/sessions/${sessionId}/messages`, { method: 'GET' }, token);

export const deleteSession = (sessionId: string, token?: string) =>
  request<void>(`/sessions/${sessionId}`, { method: 'DELETE' }, token);

// Files -----------------------------------------------------------------------

export const listFiles = (token?: string) =>
  request<FileMeta[]>('/files', { method: 'GET' }, token);

export const presignUpload = (
  body: { name: string; mime: string; doc_type?: string; week?: string | null },
  token?: string,
) =>
  request<PresignResponse>(
    '/uploads/presign',
    { method: 'POST', body: JSON.stringify(body) },
    token,
  );

/** Upload a file to the URL returned by `presignUpload`. */
export async function uploadBlob(
  upload: PresignResponse,
  data: Blob,
  token?: string,
): Promise<void> {
  // Local-dev path goes through the FastAPI backend (POST), which means we
  // need to include auth headers; R2 path is a presigned PUT direct to R2.
  const isLocal = upload.upload_url.startsWith('/uploads/local/');
  if (isLocal) {
    const headers = new Headers();
    if (token) headers.set('Authorization', `Bearer ${token}`);
    const devUser = process.env.NEXT_PUBLIC_DEV_USER;
    if (devUser && !token) headers.set('X-Dev-User', devUser);
    headers.set('Content-Type', data.type || 'application/octet-stream');
    const res = await fetch(`${API_BASE}${upload.upload_url}`, {
      method: 'POST',
      body: data,
      headers,
    });
    if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ''));
    return;
  }
  const res = await fetch(upload.upload_url, {
    method: upload.method,
    body: data,
    headers: { 'Content-Type': data.type || 'application/octet-stream' },
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ''));
}

export const processUpload = (fileId: string, token?: string) =>
  request<{ file_id: string; status: string; chunk_count: number }>(
    `/uploads/${fileId}/process`,
    { method: 'POST' },
    token,
  );

export const deleteFile = (fileId: string, token?: string) =>
  request<void>(`/files/${fileId}`, { method: 'DELETE' }, token);

// Exam (Phase 4 — endpoints stubbed until the agent wiring lands) ------------

export const generateExam = (
  body: {
    scope_type: 'file' | 'week' | 'all' | 'past_paper';
    scope_value?: string;
    num_mcq?: number;
    num_short?: number;
    difficulty?: 'easy' | 'medium' | 'hard';
  },
  token?: string,
) =>
  request<ExamPayload>(
    '/exam/generate',
    { method: 'POST', body: JSON.stringify(body) },
    token,
  );

export const submitExam = (
  examId: string,
  answers: Record<string, number | string>,
  token?: string,
) =>
  request<ExamResult>(
    `/exam/${examId}/submit`,
    { method: 'POST', body: JSON.stringify({ answers }) },
    token,
  );
