'use client';

import React from 'react';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { ChevronDown, ExternalLink, FileText, Image as ImageIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import type { CitationAnchor } from './CitationPopover';
import type { SourceHit } from '@/lib/types';
import { cn } from '@/lib/utils';

interface Props {
  sources: SourceHit[];
  /** Citation index that the parent wants to highlight + auto-open. */
  highlightedIndex?: number | null;
  /** Notify parent that a source row was clicked so it can open a popover
   *  anchored to that row. ``idx`` is 1-based (S1, S2 …). */
  onSelectSource?: (idx: number, anchor: CitationAnchor) => void;
}

function libraryHref(source: SourceHit): Route {
  // ``as Route`` is required because typedRoutes can't statically verify
  // a query-string built at runtime — the library route exists, we're
  // just appending a filter param the page reads via useSearchParams.
  const params = new URLSearchParams();
  if (source.project_id) params.set('project', source.project_id);
  if (source.folder_id) params.set('folder', source.folder_id);
  if (source.file_id) params.set('file_id', source.file_id);
  else if (source.source) params.set('file', source.source);
  const q = params.toString() ? `?${params.toString()}` : '';
  return (`/library${q}`) as Route;
}

export function SourcePanel({ sources, highlightedIndex = null, onSelectSource }: Props) {
  const [open, setOpen] = useState<boolean>(highlightedIndex !== null);
  useEffect(() => {
    if (highlightedIndex !== null) setOpen(true);
  }, [highlightedIndex]);
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
              const idx1 = i + 1;
              return (
                <li
                  key={i}
                  id={`source-${idx1}`}
                  className={cn(
                    'scroll-mt-16 transition',
                    isHit && 'bg-accent/10 ring-1 ring-accent',
                  )}
                >
                  <div className="flex items-stretch">
                    <button
                      type="button"
                      onClick={(e) => {
                        if (!onSelectSource) return;
                        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
                        onSelectSource(idx1, {
                          x: r.left,
                          y: r.top,
                          width: r.width,
                          height: r.height,
                        });
                      }}
                      className={cn(
                        'group flex flex-1 flex-col gap-1 px-3 py-2 text-left transition',
                        onSelectSource
                          ? 'cursor-pointer hover:bg-parchment/60'
                          : 'cursor-default',
                      )}
                      aria-label={`Open source S${idx1}`}
                    >
                      <div className="flex items-center gap-2 text-ink">
                        <span
                          className={cn(
                            'inline-flex h-5 min-w-[1.4rem] items-center justify-center rounded-full px-1.5 text-[10px] font-bold transition',
                            isHit
                              ? 'bg-accent text-parchment ring-2 ring-accent/40'
                              : 'bg-parchment text-ink-muted ring-1 ring-parchment-warm group-hover:bg-accent/15 group-hover:text-accent',
                          )}
                        >
                          {idx1}
                        </span>
                        <Icon className="h-3.5 w-3.5 text-accent" />
                        <span className="truncate font-medium" title={s.source ?? ''}>
                          {s.source ?? 'unknown'}
                        </span>
                        {s.week && <span className="text-ink-soft">· {s.week}</span>}
                      </div>
                      <p className="line-clamp-3 text-ink-muted">{s.snippet}</p>
                    </button>
                    {(s.file_id || s.source) && (
                      <Link
                        href={libraryHref(s)}
                        prefetch={false}
                        title="Open in library"
                        aria-label={`Open ${s.source} in library`}
                        className="flex items-center justify-center border-l border-parchment-warm/70 px-3 text-ink-soft transition hover:bg-parchment/60 hover:text-accent"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </Link>
                    )}
                  </div>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
