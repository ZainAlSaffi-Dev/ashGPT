'use client';

import { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

import { CitationBadge } from './CitationBadge';
import { useCitationCtx } from './citation-context';
import { rehypeCitations } from '@/lib/rehype-citations';

interface Props {
  bodyMd: string;
}

/**
 * Markdown body extracted from ChatMessage so React.memo can short-circuit
 * the hast re-walk + rehype pass on every popover toggle. The body only
 * needs to re-render when the streamed markdown changes. Per-chip active
 * styling is read from CitationContext inside <CitationChip /> so context
 * value changes still update chip visuals without re-running rehype.
 */
function ChatMessageBodyInner({ bodyMd }: Props) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight, rehypeCitations]}
      components={{
        // ``cite`` is emitted by rehypeCitations for ``[S#]`` tokens.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        cite: ({ node }: any) => <CitationChip node={node} />,
        // ``mark`` is emitted for ``[external]`` tokens.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        mark: ({ node, ...props }: any) => {
          if (node?.properties?.['data-external'] !== 'true') {
            return <mark {...props} />;
          }
          return (
            <span className="mr-1 inline-flex items-center rounded bg-red-100 px-1.5 py-0.5 text-[0.7em] font-semibold uppercase tracking-wide text-red-700 ring-1 ring-red-200">
              outside sources
            </span>
          );
        },
      }}
    >
      {bodyMd || ' '}
    </ReactMarkdown>
  );
}

export const ChatMessageBody = memo(ChatMessageBodyInner, (prev, next) => prev.bodyMd === next.bodyMd);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CitationChip({ node }: { node: any }) {
  const ctx = useCitationCtx();
  const idx = Number(node?.properties?.['data-source-index'] ?? 0);
  const occurrence: string = node?.properties?.['data-cite-occurrence'] ?? `${idx}-0`;
  const context: string = node?.properties?.['data-cite-context'] ?? '';
  const isPinned = ctx.pinnedOccurrence === occurrence;
  const isHovered = !isPinned && ctx.hoveredOccurrence === occurrence;

  const anchorFromEl = (el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, width: r.width, height: r.height };
  };

  return (
    <CitationBadge
      index={idx}
      occurrenceId={occurrence}
      active={isPinned}
      hovered={isHovered}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        ctx.togglePin({
          occurrence,
          idx,
          anchor: anchorFromEl(e.currentTarget as HTMLElement),
          context,
        });
      }}
      onPointerEnter={(e) => {
        if (ctx.pinnedOccurrence) return;
        ctx.scheduleHoverOpen({
          occurrence,
          idx,
          anchor: anchorFromEl(e.currentTarget as HTMLElement),
          context,
        });
      }}
      onPointerLeave={() => {
        if (ctx.pinnedOccurrence) return;
        ctx.scheduleHoverClose();
      }}
      onFocus={(e) => {
        if (ctx.pinnedOccurrence) return;
        ctx.scheduleHoverOpen({
          occurrence,
          idx,
          anchor: anchorFromEl(e.currentTarget as HTMLElement),
          context,
        });
      }}
      onBlur={() => {
        if (ctx.pinnedOccurrence) return;
        ctx.scheduleHoverClose();
      }}
    />
  );
}
