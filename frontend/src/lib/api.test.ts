import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  AuthNotReadyError,
  listSessions,
  setTokenProvider,
  withAuth,
} from './api';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  setTokenProvider(null);
  vi.useRealTimers();
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
