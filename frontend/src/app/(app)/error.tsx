'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/Button';

/** Friendly error boundary for the authed app shell.
 *
 *  The historical bug was a fresh-sign-in race where Clerk's token wasn't
 *  yet in memory; the server replied 401 and Next rendered a generic
 *  "server side error" page. We now (a) gate every authed query on
 *  ``useAuthReady``, and (b) catch anything that still slips through with
 *  this boundary so the user sees "Reconnecting…" + a retry button instead
 *  of a stack-trace-flavoured panic page.
 */
export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const message = error?.message ?? '';
  const looksLikeAuth =
    /401|AuthNotReady|unauthorized|unauthenticated/i.test(message) ||
    error?.name === 'AuthNotReadyError';

  const [retrying, setRetrying] = useState(false);

  // Auto-retry once for the auth race so most users never see this UI at
  // all — they just get a tiny "reconnecting" flicker.
  useEffect(() => {
    if (!looksLikeAuth) return;
    const t = setTimeout(() => {
      setRetrying(true);
      reset();
    }, 800);
    return () => clearTimeout(t);
  }, [looksLikeAuth, reset]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="grid min-h-[60vh] place-items-center px-6"
    >
      <div className="max-w-md rounded-xl border border-parchment-warm bg-parchment p-8 text-center shadow-sm">
        <div className="mx-auto inline-flex h-10 w-10 items-center justify-center rounded-full bg-accent/10 text-accent">
          {looksLikeAuth ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <RefreshCw className="h-5 w-5" />
          )}
        </div>
        <h1 className="mt-4 font-serif text-xl text-ink">
          {looksLikeAuth ? 'Reconnecting…' : 'Something went wrong'}
        </h1>
        <p className="mt-2 text-sm text-ink-muted">
          {looksLikeAuth
            ? 'Finalising your session. This usually clears in a second.'
            : 'We hit a snag rendering this page. You can try again or head back home.'}
        </p>
        {!looksLikeAuth && (
          <div className="mt-6 flex justify-center gap-2">
            <Button size="sm" onClick={reset} disabled={retrying}>
              Try again
            </Button>
            <Button variant="secondary" size="sm" asChild>
              <a href="/chat">Go to chat</a>
            </Button>
          </div>
        )}
      </div>
    </motion.div>
  );
}
