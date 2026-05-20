import { describe, expect, it } from 'vitest';

import { findHighlightSpan, highlightSnippet } from './citation-highlight';

describe('findHighlightSpan', () => {
  it('finds the longest preceding-context n-gram in the snippet', () => {
    const snippet = 'The High Court held that the duty of care extends to economic loss.';
    const context = '…in tort law, the duty of care has long been a contested boundary';
    const span = findHighlightSpan(snippet, context);
    expect(span).not.toBeNull();
    expect(snippet.slice(span!.start, span!.end).toLowerCase()).toContain('duty of care');
  });

  it('returns null on no overlap above the minimum n-gram', () => {
    expect(findHighlightSpan('apples and oranges grow on trees', 'octopus submarine quantum')).toBeNull();
  });

  it('returns null for empty inputs', () => {
    expect(findHighlightSpan('', 'anything')).toBeNull();
    expect(findHighlightSpan('anything', '')).toBeNull();
  });
});

describe('highlightSnippet', () => {
  it('splits the snippet into before/match/after around the overlap', () => {
    const snippet = 'The High Court held that the duty of care extends to economic loss.';
    const ctx = 'the duty of care has long been';
    const out = highlightSnippet(snippet, ctx);
    expect(out).not.toBeNull();
    expect(out!.before + out!.match + out!.after).toBe(snippet);
    expect(out!.match.toLowerCase()).toContain('duty of care');
  });

  it('returns null when no overlap', () => {
    expect(highlightSnippet('apples bananas oranges', 'kookaburra wattle gum')).toBeNull();
  });
});
