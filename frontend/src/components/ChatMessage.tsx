'use client';

import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

import { CitationPopover } from './CitationPopover';
import { MermaidRenderer } from './MermaidRenderer';
import { SourcePanel } from './SourcePanel';
import { rehypeCitations } from '@/lib/rehype-citations';
import type { ChatTurn } from '@/lib/useChat';
import { cn, extractMermaid, withoutMermaid } from '@/lib/utils';

interface Props {
  turn: ChatTurn;
}

export function ChatMessage({ turn }: Props) {
  const isUser = turn.role === 'user';
  // Prefer the discrete `mermaid` field; fall back to inline-fence parsing.
  const mermaid = turn.mermaid || extractMermaid(turn.content);
  const bodyMd = turn.mermaid ? turn.content : withoutMermaid(turn.content);
  // Track the citation the user most recently clicked so SourcePanel opens
  // and highlights the matching row.
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);
  // Floating popover anchored on the citation button. ``idx`` is 1-based
  // (S1, S2, …) so it can drive both the SourcePanel highlight (idx-1) and
  // the popover header label directly.
  const [popover, setPopover] = useState<{
    idx: number;
    anchor: { x: number; y: number; width: number };
  } | null>(null);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      className={cn(
        'rounded-2xl px-4 py-3',
        // Assistant bubbles stretch wider for readable prose; user bubbles
        // stay narrow so the eye can follow the back-and-forth.
        isUser
          ? 'ml-auto max-w-[80%] bg-accent text-parchment shadow-sm'
          : 'mr-auto w-full bg-parchment text-ink ring-1 ring-parchment-warm',
      )}
    >
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight, rehypeCitations]}
          components={{
            // ``cite`` is emitted by rehypeCitations for ``[S#]`` tokens —
            // render as a small clickable badge that highlights the matching
            // entry in the SourcePanel below.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            cite: ({ node, ...props }: any) => {
              const idx = Number(node?.properties?.['data-source-index'] ?? 0);
              return (
                <button
                  type="button"
                  data-source-index={idx}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setHighlightedSource(idx - 1);
                    // Click a second time on the same chip → close. Click
                    // a different chip → swap to that one.
                    if (popover && popover.idx === idx) {
                      setPopover(null);
                      return;
                    }
                    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                    setPopover({
                      idx,
                      anchor: {
                        x: rect.left,
                        y: rect.bottom,
                        width: rect.width,
                      },
                    });
                  }}
                  className={cn(
                    'not-italic mx-0.5 inline-flex items-center rounded px-1.5 py-0.5 align-baseline text-[0.7em] font-semibold transition',
                    popover?.idx === idx
                      ? 'bg-accent text-parchment'
                      : 'bg-accent/15 text-accent hover:bg-accent hover:text-parchment',
                  )}
                  title={`Show source S${idx}`}
                >
                  {props.children}
                </button>
              );
            },
            // ``mark`` is emitted for ``[external]`` tokens — show as a
            // visible warning chip so the student knows the next claim is
            // not in their library.
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
        {turn.streaming && (
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block w-[0.5ch] animate-[blink_1s_steps(1)_infinite] text-ink-muted"
          >
            ▍
          </span>
        )}
      </div>
      {mermaid && <MermaidRenderer diagram={mermaid} />}
      {turn.irac && (
        <details className="mt-3 rounded border border-parchment-warm bg-parchment-warm/40 text-xs">
          <summary className="cursor-pointer px-3 py-2 font-medium text-ink">IRAC analysis</summary>
          <pre className="whitespace-pre-wrap px-3 py-2 text-ink-muted">{turn.irac}</pre>
        </details>
      )}
      {turn.sources && (
        <SourcePanel sources={turn.sources} highlightedIndex={highlightedSource} />
      )}
      {turn.verification && turn.verification.all_supported === false && (
        <p className="mt-2 text-xs text-accent">
          ⚠ Some citations could not be verified in retrieved sources.
        </p>
      )}
      {turn.verification &&
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (turn.verification as any).used_external_knowledge && (
          <p className="mt-2 text-xs text-red-700">
            ⚠ This answer leaned on knowledge outside your indexed library.
            Treat the marked claims as background, not authoritative.
          </p>
        )}
      {turn.historyOverflow &&
        (turn.historyOverflow.dropped_turns > 0 ||
          turn.historyOverflow.truncated_messages > 0) && (
          <p className="mt-2 text-[11px] text-ink-soft">
            ⓘ Trimmed older context to fit ({turn.historyOverflow.dropped_turns} turn(s) dropped,{' '}
            {turn.historyOverflow.truncated_messages} message(s) truncated). Start a new chat for a fresh window.
          </p>
        )}
      {turn.latency_ms != null && !turn.streaming && (
        <p className="mt-2 text-[10px] uppercase tracking-wide text-ink-soft">
          {turn.intent ?? 'general'} · {turn.latency_ms} ms
        </p>
      )}
      <AnimatePresence>
        {popover && turn.sources && turn.sources[popover.idx - 1] && (
          <CitationPopover
            key={popover.idx}
            index={popover.idx}
            source={turn.sources[popover.idx - 1]}
            anchor={popover.anchor}
            onClose={() => setPopover(null)}
            onOpenInPanel={() => {
              setHighlightedSource(popover.idx - 1);
              document
                .getElementById(`source-${popover.idx}`)
                ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
              setPopover(null);
            }}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
