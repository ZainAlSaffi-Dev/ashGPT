'use client';

import React from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { ExternalLink, FileText, Image as ImageIcon, X } from 'lucide-react';

import { highlightSnippet } from '@/lib/citation-highlight';
import type { SourceHit } from '@/lib/types';
import { cn } from '@/lib/utils';

export interface CitationAnchor {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Props {
  source: SourceHit;
  index: number;
  /** ``hover`` = transient preview that closes on mouseleave. ``pinned`` =
   *  click-to-pin, only closes on outside-click / Esc / X. */
  mode: 'hover' | 'pinned';
  anchor: CitationAnchor;
  /** Adjacent prose captured by rehypeCitations so we can highlight the
   *  overlapping span inside the cited chunk snippet. */
  context: string;
  onClose: () => void;
  onOpenInPanel: () => void;
  onPointerEnter?: () => void;
  onPointerLeave?: () => void;
}

const POPOVER_WIDTH = 320;
const POPOVER_H = 220;
const GUTTER = 12;
const MOBILE_BREAKPOINT = 640;

// Singleton viewport store: every popover subscribes to the same size
// instead of attaching its own resize listener. Avoids piling up listeners
// when many chips rapidly mount/unmount during streaming.
type ViewportSize = { w: number; h: number };
const VP_LISTENERS = new Set<(s: ViewportSize) => void>();
let VP_BOUND = false;
let VP_SIZE: ViewportSize =
  typeof window !== 'undefined'
    ? { w: window.innerWidth, h: window.innerHeight }
    : { w: 1024, h: 768 };

function bindViewport() {
  if (VP_BOUND || typeof window === 'undefined') return;
  VP_BOUND = true;
  const onResize = () => {
    VP_SIZE = { w: window.innerWidth, h: window.innerHeight };
    VP_LISTENERS.forEach((fn) => fn(VP_SIZE));
  };
  window.addEventListener('resize', onResize, { passive: true });
}

function useViewport(): ViewportSize {
  const [size, setSize] = useState<ViewportSize>(VP_SIZE);
  useEffect(() => {
    bindViewport();
    VP_LISTENERS.add(setSize);
    setSize(VP_SIZE);
    return () => {
      VP_LISTENERS.delete(setSize);
    };
  }, []);
  return size;
}

export function CitationPopover({
  source,
  index,
  mode,
  anchor,
  context,
  onClose,
  onOpenInPanel,
  onPointerEnter,
  onPointerLeave,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { w: viewportW, h: viewportH } = useViewport();
  const isMobile = viewportW < MOBILE_BREAKPOINT;
  const isPinned = mode === 'pinned';

  // Outside-click + Esc only matter for the pinned card. Hover previews
  // are owned by the parent's pointer logic.
  useEffect(() => {
    if (!isPinned) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current) return;
      const target = e.target as Element | null;
      if (!target) return;
      // Ignore clicks on the popover itself or on any citation chip — the
      // chip's onClick already handles toggle / swap; if we dismissed here
      // the click handler would just re-pin and the popover would flicker.
      if (ref.current.contains(target)) return;
      if (target.closest('[data-source-index]')) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [isPinned, onClose]);

  const Icon =
    source.doc_type === 'lecture_slide' ||
    source.doc_type === 'slide' ||
    source.doc_type === 'image'
      ? ImageIcon
      : FileText;

  const highlight = useMemo(
    () => highlightSnippet(source.snippet ?? '', context ?? ''),
    [source.snippet, context],
  );

  const place = useMemo(() => {
    let left = anchor.x + anchor.width / 2 - POPOVER_WIDTH / 2;
    left = Math.max(GUTTER, Math.min(left, viewportW - POPOVER_WIDTH - GUTTER));
    const below = anchor.y + anchor.height + 6;
    const wantsBelow = below + POPOVER_H < viewportH - GUTTER;
    const top = wantsBelow ? below : Math.max(GUTTER, anchor.y - POPOVER_H - 6);
    return { left, top, wantsBelow };
  }, [anchor, viewportW, viewportH]);

  // Portal so ``position: fixed`` is anchored to the viewport, not to the
  // nearest ``transform``ed ancestor — ChatMessage's motion.div applies a
  // transform during entrance, which would otherwise re-root fixed
  // children and push the popover offscreen.
  if (typeof document === 'undefined') return null;

  // Mobile pinned → bottom sheet. Touch hover is unreliable so we only
  // render the sheet variant for pinned interaction.
  if (isMobile && isPinned) {
    return createPortal(
      <AnimatePresence>
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-50 bg-ink/30"
          onClick={onClose}
          aria-hidden="true"
        />
        <motion.div
          key="sheet"
          ref={ref}
          initial={{ y: '100%' }}
          animate={{ y: 0 }}
          exit={{ y: '100%' }}
          transition={{ type: 'spring', stiffness: 360, damping: 36 }}
          className="fixed inset-x-0 bottom-0 z-50 rounded-t-2xl border-t border-parchment-warm bg-parchment text-sm shadow-2xl"
          role="dialog"
          aria-label={`Source S${index}`}
        >
          <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-parchment-warm" aria-hidden="true" />
          <PopoverContent
            source={source}
            index={index}
            Icon={Icon}
            highlight={highlight}
            onClose={onClose}
            onOpenInPanel={onOpenInPanel}
            variant="sheet"
          />
        </motion.div>
      </AnimatePresence>,
      document.body,
    );
  }

  return createPortal(
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: place.wantsBelow ? -2 : 2, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: place.wantsBelow ? -2 : 2, scale: 0.98 }}
      transition={{ duration: 0.12, ease: 'easeOut' }}
      style={{ position: 'fixed', top: place.top, left: place.left, width: POPOVER_WIDTH, zIndex: 60 }}
      className="rounded-xl border border-parchment-warm bg-parchment text-xs shadow-xl ring-1 ring-ink/5"
      role={isPinned ? 'dialog' : 'tooltip'}
      aria-label={`Source S${index}`}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
    >
      <PopoverContent
        source={source}
        index={index}
        Icon={Icon}
        highlight={highlight}
        onClose={onClose}
        onOpenInPanel={onOpenInPanel}
        variant={isPinned ? 'pinned' : 'hover'}
      />
    </motion.div>,
    document.body,
  );
}

interface ContentProps {
  source: SourceHit;
  index: number;
  Icon: typeof FileText;
  highlight: ReturnType<typeof highlightSnippet>;
  onClose: () => void;
  onOpenInPanel: () => void;
  variant: 'hover' | 'pinned' | 'sheet';
}

function PopoverContent({
  source,
  index,
  Icon,
  highlight,
  onClose,
  onOpenInPanel,
  variant,
}: ContentProps) {
  const showClose = variant !== 'hover';
  return (
    <>
      <div className="flex items-center justify-between gap-2 border-b border-parchment-warm/70 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2 text-ink">
          <span className="inline-flex h-5 min-w-[1.4rem] items-center justify-center rounded-full bg-accent px-1.5 text-[10px] font-bold text-parchment">
            {index}
          </span>
          <Icon className="h-3.5 w-3.5 shrink-0 text-accent" />
          <span className="truncate font-medium" title={source.source ?? ''}>
            {source.source ?? 'unknown'}
          </span>
        </div>
        {showClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-ink-soft transition hover:bg-parchment-warm hover:text-ink"
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>

      <div className="px-3 py-2">
        {source.week && (
          <div className="mb-1 text-[10px] uppercase tracking-wide text-ink-soft">
            {source.week}
          </div>
        )}
        <p
          className={cn(
            'whitespace-pre-wrap rounded bg-accent/10 px-2 py-1.5 leading-relaxed text-ink ring-1 ring-accent/20',
            variant === 'sheet' ? 'max-h-64 overflow-y-auto' : 'max-h-32 overflow-y-auto',
          )}
        >
          {highlight ? (
            <>
              {highlight.before}
              <mark className="rounded bg-accent/25 px-0.5 text-ink ring-1 ring-accent/40">
                {highlight.match}
              </mark>
              {highlight.after}
            </>
          ) : (
            source.snippet || '(no snippet)'
          )}
        </p>
      </div>

      <div className="flex items-center justify-between border-t border-parchment-warm/70 px-3 py-2">
        <span className="text-[10px] text-ink-soft">
          {variant === 'hover' ? 'Click to pin' : 'Cited chunk from your library'}
        </span>
        <button
          type="button"
          onClick={onOpenInPanel}
          className="inline-flex items-center gap-1 rounded text-[11px] font-medium text-accent transition hover:text-accent/80"
        >
          View in sources
          <ExternalLink className="h-3 w-3" />
        </button>
      </div>
    </>
  );
}
