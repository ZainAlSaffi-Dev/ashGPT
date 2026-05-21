import React from 'react';
import { render, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { AppShell } from './AppShell';

let pathname = '/library';
const getToken = vi.fn();

vi.mock('next/navigation', () => ({
  usePathname: () => pathname,
}));

vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({ getToken }),
  UserButton: () => <div data-testid="user-button" />,
}));

vi.mock('./Sidebar', () => ({
  Sidebar: () => <nav data-testid="sidebar" />,
}));

describe('AppShell', () => {
  beforeEach(() => {
    pathname = '/library';
    getToken.mockReset();
    getToken.mockResolvedValue('token');
  });

  it('resets the app scroll container when the route path changes', async () => {
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollTo', {
      configurable: true,
      value: scrollTo,
    });

    const { rerender } = render(
      <AppShell>
        <div>Library overview</div>
      </AppShell>,
    );

    await waitFor(() => expect(scrollTo).toHaveBeenCalled());
    scrollTo.mockClear();

    pathname = '/library/project-1';
    rerender(
      <AppShell>
        <div>Project workspace</div>
      </AppShell>,
    );

    await waitFor(() =>
      expect(scrollTo).toHaveBeenCalledWith({ top: 0, left: 0, behavior: 'auto' }),
    );
  });
});
