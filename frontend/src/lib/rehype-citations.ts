/**
 * Tiny rehype plugin: turns inline ``[S1]``, ``[S12]``, ``[external]``
 * tokens in answer text into HTML elements the chat renderer can style.
 *
 *   [S1]       → <cite data-source-index="1">[S1]</cite>
 *   [external] → <mark data-external="true">[external]</mark>
 *
 * The chat ``components`` mapping in ChatMessage picks those up and renders
 * clickable badges + warning chips.
 */

import { visit } from 'unist-util-visit';
import type { Plugin } from 'unified';
import type { Root, Text, Element } from 'hast';

const CITE_RE = /\[S(\d+)\]/g;
const EXTERNAL_RE = /\[external\]/gi;

interface TextParent {
  type: 'element' | 'root';
  children: (Text | Element)[];
}

function splitTextNode(node: Text): (Text | Element)[] {
  const original = node.value;
  if (!CITE_RE.test(original) && !EXTERNAL_RE.test(original)) {
    CITE_RE.lastIndex = 0;
    EXTERNAL_RE.lastIndex = 0;
    return [node];
  }
  CITE_RE.lastIndex = 0;
  EXTERNAL_RE.lastIndex = 0;

  // Single regex sweep over both token shapes to keep ordering intact.
  const combined = /\[S(\d+)\]|\[external\]/gi;
  const out: (Text | Element)[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = combined.exec(original))) {
    if (m.index > last) {
      out.push({ type: 'text', value: original.slice(last, m.index) } as Text);
    }
    if (m[0].toLowerCase() === '[external]') {
      out.push({
        type: 'element',
        tagName: 'mark',
        properties: { 'data-external': 'true' },
        children: [{ type: 'text', value: '[external]' } as Text],
      } as Element);
    } else {
      out.push({
        type: 'element',
        tagName: 'cite',
        properties: { 'data-source-index': m[1] },
        children: [{ type: 'text', value: `[S${m[1]}]` } as Text],
      } as Element);
    }
    last = m.index + m[0].length;
  }
  if (last < original.length) {
    out.push({ type: 'text', value: original.slice(last) } as Text);
  }
  return out;
}

export const rehypeCitations: Plugin<[], Root> = () => (tree) => {
  visit(tree, 'text', (node: Text, index, parent) => {
    if (!parent || typeof index !== 'number') return;
    // Don't recurse into <code>/<pre> blocks — citations only matter in prose.
    if ('tagName' in parent && (parent.tagName === 'code' || parent.tagName === 'pre')) {
      return;
    }
    const replacement = splitTextNode(node);
    if (replacement.length === 1 && replacement[0] === node) return;
    (parent as TextParent).children.splice(index, 1, ...replacement);
  });
};
