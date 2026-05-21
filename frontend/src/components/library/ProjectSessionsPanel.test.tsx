import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ProjectSessionsPanelContent } from './ProjectSessionsPanel';
import type { SessionSummary } from '@/lib/types';

const mocks = vi.hoisted(() => ({
  createSession: vi.fn(),
  getToken: vi.fn(),
  push: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mocks.push }),
}));

vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({ getToken: mocks.getToken }),
}));

vi.mock('@/lib/api', async () => {
  const actual = await vi.importActual<typeof import('@/lib/api')>('@/lib/api');
  return {
    ...actual,
    createSession: mocks.createSession,
  };
});

function session(id: string, title: string): SessionSummary {
  return {
    id,
    title,
    project_id: 'project-1',
    folder_id: null,
    scope: { type: 'project', project_id: 'project-1' },
    created_at: '2026-05-21T08:00:00.000Z',
    updated_at: '2026-05-21T09:30:00.000Z',
  };
}

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe('ProjectSessionsPanelContent', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.createSession.mockResolvedValue({
      id: 'new-session',
      title: 'New subject chat',
      project_id: 'project-1',
      folder_id: null,
      scope: { type: 'project', project_id: 'project-1' },
      created_at: '2026-05-21T09:30:00.000Z',
      updated_at: '2026-05-21T09:30:00.000Z',
    });
    mocks.getToken.mockResolvedValue('token-1');
    mocks.push.mockClear();
  });

  it('links to recent subject chats and creates a scoped session for new chat', async () => {
    renderWithClient(
      <ProjectSessionsPanelContent
        projectId="project-1"
        sessions={[session('s1', 'Equity tracing'), session('s2', 'Fiduciary duties')]}
        isLoading={false}
      />,
    );

    expect(screen.getByRole('link', { name: /equity tracing/i }).getAttribute('href')).toBe(
      '/chat/s1',
    );
    expect(screen.getByRole('link', { name: /fiduciary duties/i }).getAttribute('href')).toBe(
      '/chat/s2',
    );

    fireEvent.click(screen.getByRole('button', { name: /new subject chat/i }));

    await waitFor(() =>
      expect(mocks.createSession).toHaveBeenCalledWith('New subject chat', 'token-1', {
        projectId: 'project-1',
        folderId: null,
        scope: { type: 'project', project_id: 'project-1' },
      }),
    );
    expect(mocks.push).toHaveBeenCalledWith('/chat/new-session');
  });

  it('shows an empty state when the subject has no chats', () => {
    renderWithClient(
      <ProjectSessionsPanelContent projectId="project-1" sessions={[]} isLoading={false} />,
    );

    expect(screen.getByText('No subject chats yet')).toBeTruthy();
    expect(screen.getByText(/stay attached to this workspace/i)).toBeTruthy();
  });
});
