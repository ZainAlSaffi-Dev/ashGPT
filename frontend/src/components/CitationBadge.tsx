'use client';

import { motion } from 'framer-motion';
import React from 'react';
import { forwardRef, type CSSProperties } from 'react';

import { cn } from '@/lib/utils';

interface Props {
  index: number;
  active?: boolean;
  hovered?: boolean;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onPointerEnter?: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerLeave?: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onFocus?: (e: React.FocusEvent<HTMLButtonElement>) => void;
  onBlur?: (e: React.FocusEvent<HTMLButtonElement>) => void;
  occurrenceId?: string;
  style?: CSSProperties;
}

/**
 * Gemini-style numbered citation chip. Small circular badge, accent ring
 * when active, subtle hover lift. Renders inline inside prose so the chip
 * sits cleanly next to the word it follows.
 */
export const CitationBadge = forwardRef<HTMLButtonElement, Props>(function CitationBadge(
  {
    index,
    active = false,
    hovered = false,
    onClick,
    onPointerEnter,
    onPointerLeave,
    onFocus,
    onBlur,
    occurrenceId,
    style,
  },
  ref,
) {
  return (
    <motion.button
      ref={ref}
      type="button"
      data-source-index={index}
      data-cite-occurrence={occurrenceId}
      onClick={onClick}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
      onFocus={onFocus}
      onBlur={onBlur}
      title={`Show source S${index}`}
      style={style}
      animate={{ scale: active ? [1, 1.08, 1] : 1 }}
      transition={{ duration: 0.28, ease: 'easeOut' }}
      whileHover={{ y: -1 }}
      className={cn(
        'not-italic mx-0.5 inline-flex h-[1.35em] min-w-[1.35em] items-center justify-center',
        'rounded-full px-[0.4em] align-[0.15em] text-[0.62em] font-semibold leading-none',
        'transition-colors duration-150 ease-out cursor-pointer select-none',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60',
        active
          ? 'bg-accent text-parchment ring-2 ring-accent/40 shadow-sm'
          : hovered
            ? 'bg-accent/25 text-accent ring-1 ring-accent/30'
            : 'bg-accent/12 text-accent ring-1 ring-accent/20 hover:bg-accent/25',
      )}
    >
      {index}
    </motion.button>
  );
});
