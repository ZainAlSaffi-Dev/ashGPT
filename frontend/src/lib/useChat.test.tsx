import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useChat, type ChatTurn } from './useChat';

const mocks = vi.hoisted(() => ({
  getToken: vi.fn(),
  streamChat: vi.fn(),
}));

vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({ getToken: mocks.getToken }),
}));

vi.mock('./streaming', () => ({
  streamChat: mocks.streamChat,
}));

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe('useChat', () => {
  beforeEach(() => {
    mocks.getToken.mockReset();
    mocks.streamChat.mockReset();
    mocks.getToken.mockResolvedValue('token');
    vi.stubGlobal('crypto', {
      randomUUID: vi
        .fn()
        .mockReturnValueOnce('user-id')
        .mockReturnValueOnce('assistant-id'),
    });
  });

  it('hydrates turns when messages arrive for the current session', () => {
    const initialTurns: ChatTurn[] = [
      { id: 'm1', role: 'user', content: 'Question' },
      { id: 'm2', role: 'assistant', content: 'Answer' },
    ];

    const { result, rerender } = renderHook(
      ({ turns }: { turns?: ChatTurn[] }) =>
        useChat({ initialSessionId: 's1', initialTurns: turns }),
      { initialProps: { turns: undefined } as { turns?: ChatTurn[] } },
    );

    expect(result.current.turns).toEqual([]);

    rerender({ turns: initialTurns });

    expect(result.current.turns).toEqual(initialTurns);
  });

  it('aborts an in-flight stream on unmount and suppresses route callbacks', async () => {
    const finished = deferred();
    let signal: AbortSignal | undefined;
    const onSessionCreated = vi.fn();
    mocks.streamChat.mockImplementation(async (_body, handlers, opts) => {
      signal = opts.signal;
      await finished.promise;
      handlers.onDone?.({
        session_id: 's1',
        intent: null,
        latency_ms: 1,
        final_answer: 'done',
      });
    });

    const { result, unmount } = renderHook(() =>
      useChat({ onSessionCreated }),
    );

    let sendPromise!: Promise<void>;
    await act(async () => {
      sendPromise = result.current.send('hello');
      await Promise.resolve();
    });

    await waitFor(() => expect(mocks.streamChat).toHaveBeenCalledTimes(1));
    expect(signal?.aborted).toBe(false);

    unmount();
    expect(signal?.aborted).toBe(true);

    finished.resolve();
    await act(async () => {
      await sendPromise;
    });

    expect(onSessionCreated).not.toHaveBeenCalled();
  });
});
