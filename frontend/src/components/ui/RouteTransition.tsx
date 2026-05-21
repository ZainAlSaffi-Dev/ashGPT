'use client';

import { usePathname } from 'next/navigation';
import { AnimatePresence, motion } from 'framer-motion';

/** Cross-fade between routes within the (app) shell.
 *
 *  Tuned for *perceived* speed: ``mode="sync"`` so the new page paints
 *  immediately and the old one fades over it (no exit-then-enter wait),
 *  opacity-only (no Y translate that triggers a paint cost), 90ms total.
 *  Net effect: navigation feels instant; the brief fade just hides the
 *  hydration flicker without adding latency.
 */
export function RouteTransition({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <AnimatePresence mode="sync" initial={false}>
      <motion.div
        key={pathname}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.09, ease: 'linear' }}
        className="h-full"
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
