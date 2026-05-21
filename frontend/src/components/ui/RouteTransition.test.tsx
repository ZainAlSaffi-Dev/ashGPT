import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { RouteTransition } from './RouteTransition';

let pathname = '/library';

vi.mock('next/navigation', () => ({
  usePathname: () => pathname,
}));

describe('RouteTransition', () => {
  it('replaces the outgoing route immediately instead of stacking pages', () => {
    const { rerender } = render(
      <RouteTransition>
        <div>Library overview</div>
      </RouteTransition>,
    );

    expect(screen.getByText('Library overview')).toBeTruthy();

    pathname = '/library/project-1';
    rerender(
      <RouteTransition>
        <div>Project workspace</div>
      </RouteTransition>,
    );

    expect(screen.queryByText('Library overview')).toBeNull();
    expect(screen.getByText('Project workspace')).toBeTruthy();
  });
});
