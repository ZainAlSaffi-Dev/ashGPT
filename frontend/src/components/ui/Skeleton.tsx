'use client';

import { motion } from 'framer-motion';

import { cn } from '@/lib/utils';

/** Shimmer-loading placeholder. Drop-in for any block element while data
 *  fetches; the animated gradient telegraphs "incoming content" without
 *  the layout shift / flicker of plain spinner text. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-md bg-parchment-warm/70',
        className,
      )}
      aria-hidden="true"
    >
      <motion.div
        className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-parchment to-transparent"
        animate={{ x: ['-100%', '100%'] }}
        transition={{ repeat: Infinity, duration: 1.4, ease: 'linear' }}
      />
    </div>
  );
}

/** Stacked skeleton bars — used for sidebar session list and library rows. */
export function SkeletonList({ rows = 4, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-7 w-full" />
      ))}
    </div>
  );
}
