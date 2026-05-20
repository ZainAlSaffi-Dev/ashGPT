import { describe, expect, it } from 'vitest';

import { cn, extractMermaid, withoutMermaid } from './utils';

describe('cn', () => {
  it('merges class names', () => {
    expect(cn('a', 'b')).toBe('a b');
  });
  it('dedupes tailwind conflicts', () => {
    // tailwind-merge keeps the latter conflicting class.
    expect(cn('px-2 px-4')).toBe('px-4');
  });
  it('ignores falsy', () => {
    expect(cn('a', false && 'b', null, undefined, 'c')).toBe('a c');
  });
});

describe('extractMermaid', () => {
  it('pulls a fenced diagram out of markdown', () => {
    const text = 'Some intro.\n\n```mermaid\nflowchart TD\nA-->B\n```\n\nConclusion.';
    expect(extractMermaid(text)).toBe('flowchart TD\nA-->B');
  });

  it('returns null when there is no diagram', () => {
    expect(extractMermaid('just words')).toBeNull();
  });
});

describe('withoutMermaid', () => {
  it('strips the diagram block', () => {
    const text = 'Intro.\n\n```mermaid\nflowchart TD\nA-->B\n```\n\nOutro.';
    expect(withoutMermaid(text)).toBe('Intro.\n\n\n\nOutro.');
  });
});
