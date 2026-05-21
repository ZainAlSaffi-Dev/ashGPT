/**
 * Tiny rehype plugin: turns inline ``[S1]``, ``[S12]``, ``[external]``
 * tokens in answer text into HTML elements the chat renderer can style.
 *
 *   [S1]       → <cite data-source-index="1"
 *                       data-cite-occurrence="1-0"
 *                       data-cite-context="…preceding prose…">[S1]</cite>
 *   [external] → <mark data-external="true">[external]</mark>
 *
 * ``data-cite-occurrence`` is a per-render counter so duplicate ``[S1]``s
 * stay individually addressable. ``data-cite-context`` carries a short
 * window of preceding prose so the popover can highlight the overlap with
 * the cited chunk snippet without round-tripping to the backend.
 */

import { visit } from 'unist-util-visit';
import type { Root, Text, Element } from 'hast';

const CITE_RE = /\[S(\d+)\]/g;
const EXTERNAL_RE = /\[external\]/gi;

interface TextParent {
  type: 'element' | 'root';
  children: (Text | Element)[];
}

interface SplitCtx {
  occurrencesByIdx: Map<number, number>;
  precedingText: string;
}

function pushPreceding(ctx: SplitCtx, chunk: string) {
  ctx.precedingText = (ctx.precedingText + chunk).slice(-400);
}

function splitTextNode(node: Text, ctx: SplitCtx): (Text | Element)[] {
  const original = node.value;
  if (!CITE_RE.test(original) && !EXTERNAL_RE.test(original)) {
    CITE_RE.lastIndex = 0;
    EXTERNAL_RE.lastIndex = 0;
    pushPreceding(ctx, original);
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
      const chunk = original.slice(last, m.index);
      out.push({ type: 'text', value: chunk } as Text);
      pushPreceding(ctx, chunk);
    }
    if (m[0].toLowerCase() === '[external]') {
      out.push({
        type: 'element',
        tagName: 'mark',
        properties: { 'data-external': 'true' },
        children: [{ type: 'text', value: '[external]' } as Text],
      } as Element);
    } else {
      const idx = Number(m[1]);
      const seen = ctx.occurrencesByIdx.get(idx) ?? 0;
      ctx.occurrencesByIdx.set(idx, seen + 1);
      out.push({
        type: 'element',
        tagName: 'cite',
        properties: {
          'data-source-index': String(idx),
          'data-cite-occurrence': `${idx}-${seen}`,
          'data-cite-context': ctx.precedingText.slice(-200),
        },
        children: [{ type: 'text', value: `[S${idx}]` } as Text],
      } as Element);
    }
    last = m.index + m[0].length;
  }
  if (last < original.length) {
    const tail = original.slice(last);
    out.push({ type: 'text', value: tail } as Text);
    pushPreceding(ctx, tail);
  }
  return out;
}

type RehypeTransformer = (tree: Root) => void;

// Plain factory signature (no ``unified`` type import — it's a transitive
// dependency and resolves at runtime, but the type isn't in package.json so
// strict ``pnpm install`` on CF Pages can't find it during the TS build).
export const rehypeCitations: () => RehypeTransformer = () => (tree) => {
  // Fresh per-render context: counters reset every time react-markdown
  // re-renders (every streamed token), so the same occurrence ids stay
  // stable across re-renders of the same prose.
  const ctx: SplitCtx = { occurrencesByIdx: new Map(), precedingText: '' };
  visit(tree, 'text', (node: Text, index, parent) => {
    if (!parent || typeof index !== 'number') return;
    // Don't recurse into <code>/<pre> blocks — citations only matter in prose.
    if ('tagName' in parent && (parent.tagName === 'code' || parent.tagName === 'pre')) {
      return;
    }
    const replacement = splitTextNode(node, ctx);
    if (replacement.length === 1 && replacement[0] === node) return;
    (parent as TextParent).children.splice(index, 1, ...replacement);
  });
};
