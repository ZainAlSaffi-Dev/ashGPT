import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  AuthNotReadyError,
  createSession,
  getSession,
  listSessions,
  setTokenProvider,
  uploadBlob,
  withAuth,
} from './api';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  setTokenProvider(null);
  vi.useRealTimers();
});

describe('uploadBlob', () => {
  it('sends auth for same-origin API upload URLs', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 204 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await uploadBlob(
      {
        file_id: 'file-1',
        upload_url: '/uploads/blob?key=usr_demo%2Fabc%2Fnotes.md',
        blob_key: 'usr_demo/abc/notes.md',
        method: 'PUT',
      },
      new Blob(['hello'], { type: '' }),
      'token-1',
      'text/markdown',
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/uploads/blob?key=usr_demo%2Fabc%2Fnotes.md');
    expect(init?.method).toBe('PUT');
    const headers = new Headers(init?.headers);
    expect(headers.get('Authorization')).toBe('Bearer token-1');
    expect(headers.get('Content-Type')).toBe('text/markdown');
  });

  it('keeps external presigned uploads unauthenticated', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 200 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await uploadBlob(
      {
        file_id: 'file-1',
        upload_url: 'https://example.r2.cloudflarestorage.com/bucket/key',
        blob_key: 'usr_demo/abc/notes.txt',
        method: 'PUT',
      },
      new Blob(['hello'], { type: 'text/plain' }),
      'token-1',
    );

    const init = fetchMock.mock.calls[0][1];
    const headers = new Headers(init?.headers);
    expect(headers.get('Authorization')).toBeNull();
    expect(headers.get('Content-Type')).toBe('text/plain');
  });
});

describe('withAuth', () => {
  it('forwards the resolved token to the inner function', async () => {
    const inner = vi.fn(async (t: string) => `ok:${t}`);
    const result = await withAuth(async () => 'tk-1', inner);
    expect(result).toBe('ok:tk-1');
    expect(inner).toHaveBeenCalledWith('tk-1');
  });

  it('throws AuthNotReadyError if the token never resolves', async () => {
    vi.useFakeTimers();
    const inner = vi.fn();
    const promise = withAuth(
      // Never resolves — simulates Clerk session not yet hydrated.
      () => new Promise<string | null>(() => {}),
      inner,
    ).catch((e) => e);
    // Skip past the 5s internal timeout.
    await vi.advanceTimersByTimeAsync(5_100);
    const err = await promise;
    expect(err).toBeInstanceOf(AuthNotReadyError);
    expect(inner).not.toHaveBeenCalled();
  });
});

describe('request 401 replay', () => {
  beforeEach(() => {
    setTokenProvider(null);
  });

  it('refreshes the token once on 401 and retries with the new bearer', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(async () =>
        new Response('unauthorized', { status: 401 }),
      )
      .mockImplementationOnce(async () =>
        new Response(JSON.stringify([{ id: 's1', title: 'Hello', created_at: '2026-01-01', updated_at: '2026-01-01' }]), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    setTokenProvider(async () => 'fresh-token');

    const sessions = await listSessions('stale-token');
    expect(sessions).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const secondCallInit = fetchMock.mock.calls[1][1] as RequestInit;
    const headers = new Headers(secondCallInit.headers);
    expect(headers.get('Authorization')).toBe('Bearer fresh-token');
  });

  it('does not loop if the refreshed token still 401s', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('still 401', { status: 401 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    setTokenProvider(async () => 'fresh-token');

    await expect(listSessions('stale')).rejects.toThrow(/API 401/);
    // exactly one retry — not infinite.
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('makes a single request when no token provider is registered', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('nope', { status: 401 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    setTokenProvider(null);
    await expect(listSessions('stale')).rejects.toThrow(/API 401/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('scoped sessions API', () => {
  it('creates a project-scoped session with scope metadata', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 's1',
          title: 'New subject chat',
          project_id: 'p1',
          folder_id: null,
          scope: { type: 'project', project_id: 'p1' },
          created_at: '2026-01-01',
          updated_at: '2026-01-01',
        }),
        { status: 201, headers: { 'content-type': 'application/json' } },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const session = await createSession('New subject chat', 'token-1', {
      projectId: 'p1',
      scope: { type: 'project', project_id: 'p1' },
    });

    expect(session.project_id).toBe('p1');
    expect(session.scope).toEqual({ type: 'project', project_id: 'p1' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/sessions');
    expect(init?.method).toBe('POST');
    expect(JSON.parse(String(init?.body))).toMatchObject({
      title: 'New subject chat',
      project_id: 'p1',
      scope: { type: 'project', project_id: 'p1' },
    });
  });

  it('fetches one session for scope rehydration', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 's1',
          title: 'Contracts',
          project_id: 'p1',
          folder_id: null,
          scope: { type: 'project', project_id: 'p1' },
          created_at: '2026-01-01',
          updated_at: '2026-01-01',
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const session = await getSession('s1', 'token-1');

    expect(session.id).toBe('s1');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/sessions/s1');
    expect(init?.method).toBe('GET');
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer token-1');
  });
});
