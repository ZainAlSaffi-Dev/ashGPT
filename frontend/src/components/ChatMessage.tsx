'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { ChatMessageBody } from './ChatMessageBody';
import { CitationContext, type CitationTarget } from './citation-context';
import { CitationPopover, type CitationAnchor } from './CitationPopover';
import { MermaidRenderer } from './MermaidRenderer';
import { SourcePanel } from './SourcePanel';
import { rehypeCitations } from '@/lib/rehype-citations';
import type { ChatTurn } from '@/lib/useChat';
import { cn, extractMermaid, withoutMermaid } from '@/lib/utils';

interface Props {
  turn: ChatTurn;
}

const HOVER_OPEN_DELAY = 250;
const HOVER_CLOSE_DELAY = 120;

export function ChatMessage({ turn }: Props) {
  const isUser = turn.role === 'user';
  // Prefer the discrete `mermaid` field; fall back to inline-fence parsing.
  const mermaid = turn.mermaid || extractMermaid(turn.content);
  const bodyMd = turn.mermaid ? turn.content : withoutMermaid(turn.content);

  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);
  const [pinned, setPinned] = useState<CitationTarget | null>(null);
  const [hover, setHover] = useState<CitationTarget | null>(null);

  // Open delay stops accidental cursor sweeps; close delay lets the cursor
  // travel from chip onto the card without dismissing it.
  const openTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearTimers = useCallback(() => {
    if (openTimer.current) clearTimeout(openTimer.current);
    if (closeTimer.current) clearTimeout(closeTimer.current);
    openTimer.current = null;
    closeTimer.current = null;
  }, []);
  useEffect(() => clearTimers, [clearTimers]);

  const scheduleHoverOpen = useCallback((target: CitationTarget) => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
    if (openTimer.current) clearTimeout(openTimer.current);
    openTimer.current = setTimeout(() => {
      setHover((cur) => (cur?.occurrence === target.occurrence ? cur : target));
    }, HOVER_OPEN_DELAY);
  }, []);

  const scheduleHoverClose = useCallback(() => {
    if (openTimer.current) {
      clearTimeout(openTimer.current);
      openTimer.current = null;
    }
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => {
      setHover(null);
    }, HOVER_CLOSE_DELAY);
  }, []);

  const togglePin = useCallback(
    (target: CitationTarget) => {
      clearTimers();
      setHover(null);
      setHighlightedSource(target.idx - 1);
      setPinned((cur) => (cur?.occurrence === target.occurrence ? null : target));
    },
    [clearTimers],
  );

  const ctxValue = useMemo(
    () => ({
      pinnedOccurrence: pinned?.occurrence ?? null,
      hoveredOccurrence: hover?.occurrence ?? null,
      togglePin,
      scheduleHoverOpen,
      scheduleHoverClose,
    }),
    [pinned?.occurrence, hover?.occurrence, togglePin, scheduleHoverOpen, scheduleHoverClose],
  );

  const showPinned = pinned;
  const showHover = !pinned && hover;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      className={cn(
        'rounded-2xl px-4 py-3',
        isUser
          ? 'ml-auto max-w-[80%] bg-accent text-parchment shadow-sm'
          : 'mr-auto w-full bg-parchment text-ink ring-1 ring-parchment-warm',
      )}
    >
      <CitationContext.Provider value={ctxValue}>
        <div className="prose prose-sm max-w-none">
          <ChatMessageBody bodyMd={bodyMd} />
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
          <section className="mt-3 rounded-lg border border-parchment-warm bg-parchment-warm/40 px-3 py-2">
            <h3 className="mb-1 font-serif text-sm font-semibold text-ink">
              IRAC Analysis
            </h3>
            <div className="prose prose-sm max-w-none text-ink-muted prose-headings:font-serif prose-headings:text-ink prose-strong:text-ink">
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeCitations]}>
                {turn.irac}
              </ReactMarkdown>
            </div>
          </section>
        )}
        {turn.sources && (
          <SourcePanel
            sources={turn.sources}
            highlightedIndex={highlightedSource}
            onSelectSource={(idx, anchor: CitationAnchor) => {
              clearTimers();
              setHover(null);
              setHighlightedSource(idx - 1);
              setPinned({
                occurrence: `panel-${idx}`,
                idx,
                anchor,
                context: '',
              });
            }}
          />
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
          {showPinned && turn.sources && turn.sources[showPinned.idx - 1] && (
            <CitationPopover
              key={`pin-${showPinned.occurrence}`}
              mode="pinned"
              index={showPinned.idx}
              source={turn.sources[showPinned.idx - 1]}
              anchor={showPinned.anchor}
              context={showPinned.context}
              onClose={() => setPinned(null)}
              onOpenInPanel={() => {
                setHighlightedSource(showPinned.idx - 1);
                document
                  .getElementById(`source-${showPinned.idx}`)
                  ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                setPinned(null);
              }}
            />
          )}
          {showHover && turn.sources && turn.sources[showHover.idx - 1] && (
            <CitationPopover
              key={`hov-${showHover.occurrence}`}
              mode="hover"
              index={showHover.idx}
              source={turn.sources[showHover.idx - 1]}
              anchor={showHover.anchor}
              context={showHover.context}
              onClose={() => setHover(null)}
              onOpenInPanel={() => {
                setHighlightedSource(showHover.idx - 1);
                document
                  .getElementById(`source-${showHover.idx}`)
                  ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                setHover(null);
              }}
              onPointerEnter={() => {
                if (closeTimer.current) {
                  clearTimeout(closeTimer.current);
                  closeTimer.current = null;
                }
              }}
              onPointerLeave={scheduleHoverClose}
            />
          )}
        </AnimatePresence>
      </CitationContext.Provider>
    </motion.div>
  );
}
