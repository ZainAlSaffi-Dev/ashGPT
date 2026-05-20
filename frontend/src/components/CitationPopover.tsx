'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { ExternalLink, FileText, Image as ImageIcon, X } from 'lucide-react';

import type { SourceHit } from '@/lib/types';

interface Anchor {
  x: number;
  y: number;
  width: number;
}

interface Props {
  source: SourceHit;
  index: number; // 1-based S# for display
  anchor: Anchor;
  onClose: () => void;
  onOpenInPanel: () => void;
}

const POPOVER_WIDTH = 320;
const GUTTER = 12;

export function CitationPopover({ source, index, anchor, onClose, onOpenInPanel }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click / Escape — same semantics as a small modal.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    // ``mousedown`` over ``click`` so a fresh click on a different citation
    // closes this popover before the new one opens (otherwise the parent's
    // click handler can't distinguish "swap" from "close").
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  const Icon =
    source.doc_type === 'lecture_slide' ||
    source.doc_type === 'slide' ||
    source.doc_type === 'image'
      ? ImageIcon
      : FileText;

  // Position: try to centre under the citation but keep the panel fully
  // on-screen with a small gutter. ``position: fixed`` so the popover
  // floats above the chat scroll container without getting clipped.
  const viewportW = typeof window !== 'undefined' ? window.innerWidth : 1024;
  const viewportH = typeof window !== 'undefined' ? window.innerHeight : 768;
  let left = anchor.x + anchor.width / 2 - POPOVER_WIDTH / 2;
  left = Math.max(GUTTER, Math.min(left, viewportW - POPOVER_WIDTH - GUTTER));
  // Default below the citation. If that would overflow the viewport, flip
  // above (popover height ~180px; treat as fixed budget).
  const POPOVER_H = 220;
  const wantsBelow = anchor.y + 6 + POPOVER_H < viewportH - GUTTER;
  const top = wantsBelow ? anchor.y + 6 : Math.max(GUTTER, anchor.y - POPOVER_H - 6);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: wantsBelow ? -4 : 4, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.12, ease: 'easeOut' }}
      style={{ position: 'fixed', top, left, width: POPOVER_WIDTH, zIndex: 60 }}
      className="rounded-xl border border-parchment-warm bg-parchment text-xs shadow-xl ring-1 ring-ink/5"
      role="dialog"
      aria-label={`Source S${index}`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-parchment-warm/70 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2 text-ink">
          <span className="inline-flex h-5 w-7 items-center justify-center rounded bg-accent text-[10px] font-bold text-parchment">
            S{index}
          </span>
          <Icon className="h-3.5 w-3.5 shrink-0 text-accent" />
          <span className="truncate font-medium" title={source.source ?? ''}>
            {source.source ?? 'unknown'}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="rounded p-1 text-ink-soft transition hover:bg-parchment-warm hover:text-ink"
        >
          <X className="h-3 w-3" />
        </button>
      </div>

      <div className="px-3 py-2">
        {source.week && (
          <div className="mb-1 text-[10px] uppercase tracking-wide text-ink-soft">
            {source.week}
          </div>
        )}
        <p className="max-h-32 overflow-y-auto whitespace-pre-wrap rounded bg-accent/10 px-2 py-1.5 leading-relaxed text-ink ring-1 ring-accent/20">
          {source.snippet || '(no snippet)'}
        </p>
      </div>

      <div className="flex items-center justify-between border-t border-parchment-warm/70 px-3 py-2">
        <span className="text-[10px] text-ink-soft">
          Cited chunk from your library
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
    </motion.div>
  );
}
