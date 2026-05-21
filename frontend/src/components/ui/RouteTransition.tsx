'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';

/** Fade in routes within the (app) shell.
 *
 *  The app shell scrolls inside its own ``main`` element, not ``window``.
 *  Keep this transition enter-only so the outgoing route cannot remain in
 *  normal document flow above the incoming route while it fades out.
 */
export function RouteTransition({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <motion.div
      key={pathname}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.09, ease: 'linear' }}
      className="min-h-full"
    >
      {children}
    </motion.div>
  );
}
