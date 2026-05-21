import { describe, expect, it } from 'vitest';

import { turnsToCachedMessages, upsertCachedSession } from './chat-cache';
import type { ChatTurn } from './useChat';

const turns: ChatTurn[] = [
  { id: 'u1', role: 'user', content: 'Explain promissory estoppel' },
  {
    id: 'a1',
    role: 'assistant',
    content: 'Promissory estoppel prevents departure from an assumption. [S1]',
    intent: 'ratio',
    sources: [{ source: 'equity.pdf', doc_type: 'note', week: '3', snippet: 'assumption' }],
    latency_ms: 123,
  },
];

describe('turnsToCachedMessages', () => {
  it('preserves streamed assistant metadata for instant route hydration', () => {
    const messages = turnsToCachedMessages(turns, '2026-05-21T00:00:00.000Z');

    expect(messages).toEqual([
      expect.objectContaining({
        id: 'u1',
        role: 'user',
        content: 'Explain promissory estoppel',
        sources: null,
        created_at: '2026-05-21T00:00:00.000Z',
      }),
      expect.objectContaining({
        id: 'a1',
        role: 'assistant',
        intent: 'ratio',
        sources: turns[1].sources,
        latency_ms: 123,
      }),
    ]);
  });
});

describe('upsertCachedSession', () => {
  it('adds a new session to the top without dropping existing history', () => {
    const sessions = upsertCachedSession(
      [{ id: 'old', title: 'Old', created_at: 'old', updated_at: 'old' }],
      'new',
      turns,
      'now',
    );

    expect(sessions.map((session) => session.id)).toEqual(['new', 'old']);
    expect(sessions[0]).toMatchObject({
      id: 'new',
      title: 'Explain promissory estoppel',
      created_at: 'now',
      updated_at: 'now',
    });
  });

  it('updates an existing session in place and moves it to the top', () => {
    const sessions = upsertCachedSession(
      [
        { id: 'a', title: 'A', created_at: 'a-created', updated_at: 'a-old' },
        { id: 'b', title: 'B', created_at: 'b-created', updated_at: 'b-old' },
      ],
      'b',
      turns,
      'now',
    );

    expect(sessions.map((session) => session.id)).toEqual(['b', 'a']);
    expect(sessions[0].created_at).toBe('b-created');
    expect(sessions[0].updated_at).toBe('now');
  });

  it('stores scoped subject metadata on the cached session row', () => {
    const sessions = upsertCachedSession(undefined, 'scoped', turns, 'now', {
      project_id: 'project-1',
      folder_id: 'folder-1',
      scope: { type: 'folder', project_id: 'project-1', folder_id: 'folder-1' },
    });

    expect(sessions[0]).toMatchObject({
      id: 'scoped',
      project_id: 'project-1',
      folder_id: 'folder-1',
      scope: { type: 'folder', project_id: 'project-1', folder_id: 'folder-1' },
    });
  });
});
