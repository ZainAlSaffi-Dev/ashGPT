'use client';

import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

/** Root-level error boundary for any unhandled server-side exception
 *  in the public route tree (landing page, sign-in modal mount, etc.).
 *
 *  Historically Clerk's RSC ``auth()`` could throw during the first
 *  request after sign-up while the JWKS cache primed; Next then
 *  rendered its bare "Application error" page with only a digest.
 *  This boundary catches that case, auto-retries once (handles the
 *  common case where the second request succeeds), and falls back to
 *  a small reload card.
 */
export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // One automatic reset attempt — Clerk's race-condition exceptions
    // clear themselves on the next render. The boundary won't loop
    // because if reset() throws again React surfaces the error.
    const t = setTimeout(reset, 600);
    return () => clearTimeout(t);
  }, [reset]);

  return (
    <main className="grid min-h-screen place-items-center bg-parchment px-6">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="max-w-md rounded-xl border border-parchment-warm bg-parchment p-8 text-center shadow-sm"
      >
        <div className="mx-auto inline-flex h-10 w-10 items-center justify-center rounded-full bg-accent/10 text-accent">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
        <h1 className="mt-4 font-serif text-xl text-ink">Reconnecting…</h1>
        <p className="mt-2 text-sm text-ink-muted">
          Finalising your session. This usually clears in a moment.
        </p>
        {error?.digest && (
          <p className="mt-3 text-[10px] text-ink-soft">ref: {error.digest}</p>
        )}
      </motion.div>
    </main>
  );
}
