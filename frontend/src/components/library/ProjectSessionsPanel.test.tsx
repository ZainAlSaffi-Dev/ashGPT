import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ProjectSessionsPanelContent } from './ProjectSessionsPanel';
import type { SessionSummary } from '@/lib/types';

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

describe('ProjectSessionsPanelContent', () => {
  it('links to recent subject chats and the scoped new-chat entry', () => {
    render(
      <ProjectSessionsPanelContent
        projectId="project-1"
        sessions={[session('s1', 'Equity tracing'), session('s2', 'Fiduciary duties')]}
        isLoading={false}
      />,
    );

    expect(screen.getByRole('link', { name: /new subject chat/i }).getAttribute('href')).toBe(
      '/chat?project=project-1',
    );
    expect(screen.getByRole('link', { name: /equity tracing/i }).getAttribute('href')).toBe(
      '/chat/s1',
    );
    expect(screen.getByRole('link', { name: /fiduciary duties/i }).getAttribute('href')).toBe(
      '/chat/s2',
    );
  });

  it('shows an empty state when the subject has no chats', () => {
    render(
      <ProjectSessionsPanelContent projectId="project-1" sessions={[]} isLoading={false} />,
    );

    expect(screen.getByText('No subject chats yet')).toBeTruthy();
    expect(screen.getByText(/stay attached to this workspace/i)).toBeTruthy();
  });
});
