/**
 * Typed fetch wrappers for the ashGPT backend.
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
  FileListScope,
  Folder,
  Message,
  PresignResponse,
  Project,
  RetrievalScope,
  SessionSummary,
} from './types';

// Resolve the API base URL with a runtime fallback. Build-time env vars
// (process.env.NEXT_PUBLIC_API_BASE) haven't been reliably plumbed through
// the Pages build pipeline, so we also detect the hosted domain and route
// directly to the edge Worker when we're on it.
export function resolveApiBase(): string {
  const buildBase = process.env.NEXT_PUBLIC_API_BASE;
  if (buildBase) return buildBase;
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    // Production custom domain → custom-domain edge worker.
    if (host === 'ashgpt.xyz' || host === 'www.ashgpt.xyz') {
      return 'https://api.ashgpt.xyz';
    }
    // Preview deploy on pages.dev → legacy workers.dev URL until the
    // worker custom domain backs preview traffic too.
    if (host.endsWith('pages.dev')) {
      return 'https://lawgpt-edge.hypersonic3692.workers.dev';
    }
  }
  return '/api';
}

const API_BASE = resolveApiBase();

export class ApiError extends Error {
  constructor(public status: number, public body: string) {
    super(`API ${status}: ${body.slice(0, 200)}`);
  }
}

/** Thrown by ``withAuth`` when Clerk's session token never resolves within the
 *  configured timeout. Callers should surface this as a "reconnecting" UI
 *  state and let TanStack retry, rather than rendering a generic error. */
export class AuthNotReadyError extends Error {
  constructor() {
    super('Auth not ready — Clerk session token unavailable');
    this.name = 'AuthNotReadyError';
  }
}

/** Hook-set callback used by ``request()`` to refresh a token on 401.
 *  Wired by ``Providers.tsx`` so the API layer can replay a single
 *  request with a fresh JWT without dragging Clerk's React hooks into
 *  every queryFn. */
type TokenProvider = (opts?: { skipCache?: boolean }) => Promise<string | null>;
let _tokenProvider: TokenProvider | null = null;
export function setTokenProvider(provider: TokenProvider | null): void {
  _tokenProvider = provider;
}

const TOKEN_TIMEOUT_MS = 5_000;
const REQUEST_TIMEOUT_MS = 15_000;
type ApiRequestInit = RequestInit & { timeoutMs?: number };

/** Await the registered token provider with a hard timeout.
 *
 *  After a fresh sign-in Clerk's in-memory token can be briefly null while
 *  the SDK hydrates; resolving with a timeout here lets callers surface a
 *  "reconnecting" state instead of hanging the request indefinitely.
 */
export async function withAuth<T>(
  getToken: TokenProvider,
  fn: (token: string) => Promise<T>,
): Promise<T> {
  const token = await Promise.race([
    getToken(),
    new Promise<null>((resolve) => setTimeout(() => resolve(null), TOKEN_TIMEOUT_MS)),
  ]);
  if (!token) throw new AuthNotReadyError();
  return fn(token);
}

async function request<T>(
  path: string,
  init: ApiRequestInit = {},
  token?: string,
  attempt: number = 0,
): Promise<T> {
  const { timeoutMs = REQUEST_TIMEOUT_MS, ...fetchInit } = init;
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const devUser = process.env.NEXT_PUBLIC_DEV_USER;
  if (devUser && !token) headers.set('X-Dev-User', devUser);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set('Content-Type', headers.get('Content-Type') ?? 'application/json');
  }

  // Apply a request timeout unless the caller already supplied a signal
  // (chat streaming wires its own AbortSignal).
  const controller = init.signal || timeoutMs <= 0 ? null : new AbortController();
  const timer = controller
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null;

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...fetchInit,
      headers,
      signal: init.signal ?? controller?.signal,
    });
  } finally {
    if (timer) clearTimeout(timer);
  }

  // One-shot replay on 401: most often a fresh sign-in race where the token
  // wasn't yet in memory. Refresh via the registered provider (skipCache)
  // and retry exactly once.
  if (res.status === 401 && attempt === 0 && _tokenProvider) {
    const fresh = await _tokenProvider({ skipCache: true }).catch(() => null);
    if (fresh && fresh !== token) {
      return request<T>(path, init, fresh, attempt + 1);
    }
  }

  if (!res.ok) {
    throw new ApiError(res.status, await res.text().catch(() => ''));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// Sessions --------------------------------------------------------------------

function qs(params: Record<string, string | null | undefined>): string {
  const out = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value) out.set(key, value);
  }
  const encoded = out.toString();
  return encoded ? `?${encoded}` : '';
}

export const listSessions = (token?: string, options: { projectId?: string | null } = {}) =>
  request<SessionSummary[]>(
    `/sessions${qs({ project_id: options.projectId })}`,
    { method: 'GET' },
    token,
  );

export const getSession = (sessionId: string, token?: string) =>
  request<SessionSummary>(`/sessions/${sessionId}`, { method: 'GET' }, token);

export const createSession = (
  title: string | null,
  token?: string,
  options: { projectId?: string | null; folderId?: string | null; scope?: RetrievalScope | null } = {},
) =>
  request<SessionSummary>(
    '/sessions',
    {
      method: 'POST',
      body: JSON.stringify({
        title,
        project_id: options.projectId,
        folder_id: options.folderId,
        scope: options.scope,
      }),
    },
    token,
  );

export const listMessages = (sessionId: string, token?: string) =>
  request<Message[]>(`/sessions/${sessionId}/messages`, { method: 'GET' }, token);

export const deleteSession = (sessionId: string, token?: string) =>
  request<void>(`/sessions/${sessionId}`, { method: 'DELETE' }, token);

// Files -----------------------------------------------------------------------

export const listProjects = (token?: string) =>
  request<Project[]>('/projects', { method: 'GET' }, token);

export const createProject = (body: { name: string; description?: string; color?: string }, token?: string) =>
  request<Project>('/projects', { method: 'POST', body: JSON.stringify(body) }, token);

export const updateProject = (
  projectId: string,
  body: { name?: string; description?: string; color?: string; archived?: boolean },
  token?: string,
) => request<Project>(`/projects/${projectId}`, { method: 'PATCH', body: JSON.stringify(body) }, token);

export const listFolders = (projectId: string, token?: string) =>
  request<Folder[]>(`/projects/${projectId}/folders`, { method: 'GET' }, token);

export const createFolder = (
  projectId: string,
  body: { name: string; parent_id?: string | null; sort_order?: number },
  token?: string,
) =>
  request<Folder>(
    `/projects/${projectId}/folders`,
    { method: 'POST', body: JSON.stringify(body) },
    token,
  );

export const updateFolder = (
  folderId: string,
  body: { name?: string; parent_id?: string | null; sort_order?: number },
  token?: string,
) => request<Folder>(`/folders/${folderId}`, { method: 'PATCH', body: JSON.stringify(body) }, token);

export const deleteFolder = (folderId: string, recursive = false, token?: string) =>
  request<void>(`/folders/${folderId}${recursive ? '?recursive=true' : ''}`, { method: 'DELETE' }, token);

export const listFiles = (token?: string, scope: FileListScope = {}) =>
  request<FileMeta[]>(
    `/files${qs({
      project_id: scope.projectId,
      folder_id: scope.folderId,
      status: scope.status,
    })}`,
    { method: 'GET' },
    token,
  );

export const presignUpload = (
  body: {
    name: string;
    mime: string;
    doc_type?: string;
    week?: string | null;
    project_id?: string | null;
    folder_id?: string | null;
  },
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
  contentType: string = data.type || 'application/octet-stream',
): Promise<void> {
  const isApiUpload =
    upload.upload_url.startsWith('/uploads/') ||
    upload.upload_url.startsWith(`${API_BASE}/uploads/`);
  if (isApiUpload) {
    const url = upload.upload_url.startsWith('/uploads/')
      ? `${API_BASE}${upload.upload_url}`
      : upload.upload_url;
    const headers = new Headers();
    if (token) headers.set('Authorization', `Bearer ${token}`);
    const devUser = process.env.NEXT_PUBLIC_DEV_USER;
    if (devUser && !token) headers.set('X-Dev-User', devUser);
    headers.set('Content-Type', contentType);
    const res = await fetch(url, {
      method: upload.method,
      body: data,
      headers,
    });
    if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ''));
    return;
  }
  const res = await fetch(upload.upload_url, {
    method: upload.method,
    body: data,
    headers: { 'Content-Type': contentType },
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ''));
}

export const processUpload = (fileId: string, token?: string) =>
  request<{ file_id: string; status: string; chunk_count: number; job_id?: string | null }>(
    `/uploads/${fileId}/process`,
    { method: 'POST', timeoutMs: 120_000 },
    token,
  );

export const deleteFile = (fileId: string, token?: string) =>
  request<void>(`/files/${fileId}`, { method: 'DELETE' }, token);

export const updateFile = (
  fileId: string,
  body: {
    name?: string;
    project_id?: string | null;
    folder_id?: string | null;
    doc_type?: string;
    week?: string | null;
  },
  token?: string,
) => request<FileMeta>(`/files/${fileId}`, { method: 'PATCH', body: JSON.stringify(body) }, token);

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

// Users -----------------------------------------------------------------------

export interface UserMe {
  id: string;
  clerk_id: string;
  email: string | null;
  onboarded_at: string | null;
  created_at: string;
}

export const getMe = (token?: string) =>
  request<UserMe>('/users/me', { method: 'GET' }, token);

export const markOnboarded = (token?: string) =>
  request<UserMe>('/users/me/onboarded', { method: 'POST' }, token);
