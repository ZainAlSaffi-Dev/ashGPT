import { describe, expect, it } from 'vitest';
import type { Root, Element } from 'hast';

import { rehypeCitations } from './rehype-citations';

/** Build a minimal hast tree wrapping a single text node and run the plugin
 *  against it. Returns the transformed root for assertions. */
function transform(text: string): Root {
  const tree: Root = {
    type: 'root',
    children: [
      {
        type: 'element',
        tagName: 'p',
        properties: {},
        children: [{ type: 'text', value: text }],
      } as Element,
    ],
  };
  // rehypeCitations is a unified Plugin; the bare invocation returns the
  // transformer. Cast through ``unknown`` because the plugin signature is
  // bound to Processor and we're driving it directly in a unit test.
  const factory = rehypeCitations as unknown as () => (t: Root) => void;
  factory()(tree);
  return tree;
}

function citeNodes(tree: Root): Element[] {
  const p = tree.children[0] as Element;
  return p.children.filter(
    (c) => c.type === 'element' && (c as Element).tagName === 'cite',
  ) as Element[];
}

describe('rehypeCitations', () => {
  it('assigns distinct occurrence ids to duplicate [S#] tokens', () => {
    const tree = transform('foo [S1] bar [S1] baz [S2]');
    const cites = citeNodes(tree);
    expect(cites).toHaveLength(3);
    expect(cites[0].properties?.['data-cite-occurrence']).toBe('1-0');
    expect(cites[1].properties?.['data-cite-occurrence']).toBe('1-1');
    expect(cites[2].properties?.['data-cite-occurrence']).toBe('2-0');
  });

  it('emits per-chip preceding context for snippet matching', () => {
    const tree = transform('the duty of care extends [S1]');
    const cite = citeNodes(tree)[0];
    expect(String(cite.properties?.['data-cite-context'])).toMatch(/duty of care/);
  });

  it('keeps preceding-context window scoped per citation', () => {
    const tree = transform('alpha bravo [S1] charlie delta [S2]');
    const cites = citeNodes(tree);
    expect(String(cites[0].properties?.['data-cite-context'])).toMatch(/alpha bravo/);
    expect(String(cites[1].properties?.['data-cite-context'])).toMatch(/charlie delta/);
  });

  it('renders [external] markers as a separate chip type', () => {
    const tree = transform('claim [external]');
    const p = tree.children[0] as Element;
    const marks = p.children.filter(
      (c) => c.type === 'element' && (c as Element).tagName === 'mark',
    ) as Element[];
    expect(marks).toHaveLength(1);
    expect(marks[0].properties?.['data-external']).toBe('true');
  });
});
