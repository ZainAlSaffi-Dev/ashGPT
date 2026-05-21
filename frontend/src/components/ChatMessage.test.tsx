import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ChatMessage } from './ChatMessage';
import type { ChatTurn } from '@/lib/useChat';

vi.mock('next/link', () => ({
  default: ({ href, children, prefetch: _prefetch, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { prefetch?: boolean }) => (
    <a href={String(href)} {...props}>
      {children}
    </a>
  ),
}));

describe('ChatMessage', () => {
  it('renders IRAC citations as interactive source chips', async () => {
    Object.defineProperty(window, 'scrollTo', {
      configurable: true,
      value: vi.fn(),
    });
    const turn: ChatTurn = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'Main answer cites title registration [S1].',
      irac: 'Issue\n\nFrazer involved a forged mortgage and registration [S2].',
      sources: [
        {
          source: 'torrens-principles.md',
          doc_type: 'note',
          week: null,
          snippet: 'Registration confers title under the Torrens system.',
        },
        {
          source: 'frazer-v-walker.md',
          doc_type: 'case',
          week: null,
          snippet: 'Frazer involved a forged mortgage and registration.',
        },
      ],
    };

    render(<ChatMessage turn={turn} />);

    fireEvent.click(screen.getByTitle('Show source S2'));

    const dialog = await screen.findByRole('dialog', { name: 'Source S2' });
    expect(within(dialog).getByText('frazer-v-walker.md')).toBeTruthy();
    expect(within(dialog).getByText(/forged mortgage and registration/i)).toBeTruthy();
  });
});
