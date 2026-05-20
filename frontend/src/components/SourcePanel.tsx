'use client';

import { useState } from 'react';
import { ChevronDown, FileText, Image as ImageIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import type { SourceHit } from '@/lib/types';
import { cn } from '@/lib/utils';

interface Props {
  sources: SourceHit[];
  /** Citation index that the parent wants to highlight + auto-open. */
  highlightedIndex?: number | null;
}

export function SourcePanel({ sources, highlightedIndex = null }: Props) {
  const [open, setOpen] = useState<boolean>(highlightedIndex !== null);
  if (!sources?.length) return null;

  return (
    <div className="mt-3 rounded-lg border border-parchment-warm bg-parchment-warm/40 text-xs text-ink-muted">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-2 font-medium text-ink"
      >
        <span>
          {sources.length} source{sources.length === 1 ? '' : 's'}
        </span>
        <ChevronDown
          className={cn(
            'h-3.5 w-3.5 transition-transform',
            open && 'rotate-180',
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
            className="divide-y divide-parchment-warm overflow-hidden"
          >
            {sources.map((s, i) => {
              const Icon =
                s.doc_type === 'lecture_slide' || s.doc_type === 'slide' || s.doc_type === 'image'
                  ? ImageIcon
                  : FileText;
              const isHit = highlightedIndex === i;
              return (
                <li
                  key={i}
                  id={`source-${i + 1}`}
                  className={cn(
                    'scroll-mt-16 px-3 py-2 transition',
                    isHit && 'bg-accent/10 ring-1 ring-accent',
                  )}
                >
                  <div className="flex items-center gap-2 text-ink">
                    <span
                      className={cn(
                        'inline-flex h-5 w-7 items-center justify-center rounded text-[10px] font-bold',
                        isHit
                          ? 'bg-accent text-parchment'
                          : 'bg-parchment text-ink-muted ring-1 ring-parchment-warm',
                      )}
                    >
                      S{i + 1}
                    </span>
                    <Icon className="h-3.5 w-3.5 text-accent" />
                    <span className="truncate font-medium" title={s.source ?? ''}>
                      {s.source ?? 'unknown'}
                    </span>
                    {s.week && <span className="text-ink-soft">· {s.week}</span>}
                  </div>
                  <p className="mt-1 text-ink-muted">{s.snippet}</p>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
