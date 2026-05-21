'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { SignInButton, SignUpButton, useAuth } from '@clerk/nextjs';
import { motion } from 'framer-motion';
import { BookMarked } from 'lucide-react';

import { Button } from '@/components/ui/Button';

/** Landing page is now fully client-rendered.
 *
 *  The previous version called Clerk's server-side ``auth()`` in an edge
 *  RSC. Right after sign-in / sign-up, the redirected request could arrive
 *  with a session token whose JWKS hadn't been cached yet — Clerk would
 *  throw inside the RSC, the App Router rendered the bare "Application
 *  error: a server-side exception has occurred" page (digest only), and
 *  the user had to refresh.
 *
 *  Moving the auth read to ``useAuth`` (client hook) sidesteps that
 *  entire race: there is no server render to fail, Clerk hydrates on the
 *  client at its own pace, and the redirect happens once we know the
 *  user's signed-in state for real.
 */
export default function LandingPage() {
  const router = useRouter();
  const { isLoaded, isSignedIn } = useAuth();

  useEffect(() => {
    if (isLoaded && isSignedIn) router.replace('/chat');
  }, [isLoaded, isSignedIn, router]);

  return (
    <main className="grid min-h-screen place-items-center bg-parchment px-6 py-16">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="max-w-xl text-center"
      >
        <motion.div
          initial={{ scale: 0.8, rotate: -10 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: 'spring', stiffness: 220, damping: 18, delay: 0.05 }}
          className="mx-auto inline-flex"
        >
          <BookMarked className="h-10 w-10 text-accent" />
        </motion.div>
        <h1 className="mt-6 font-serif text-4xl text-ink">ashGPT</h1>
        <p className="mt-4 text-ink-muted">
          Your notes, with IRAC analysis, chronology diagrams, and exam practice.
          Built for law students who want quick answers backed by their own readings.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          {isLoaded && !isSignedIn && (
            <>
              <SignUpButton mode="modal" forceRedirectUrl="/chat" fallbackRedirectUrl="/chat">
                <Button size="lg">Create account</Button>
              </SignUpButton>
              <SignInButton mode="modal" forceRedirectUrl="/chat" fallbackRedirectUrl="/chat">
                <Button size="lg" variant="secondary">
                  Sign in
                </Button>
              </SignInButton>
            </>
          )}
          {isLoaded && isSignedIn && (
            <Button size="lg" asChild>
              <Link href="/chat">Open the app</Link>
            </Button>
          )}
        </div>
        {/* While Clerk is hydrating (or while we're mid-redirect for a
            signed-in user) show a calm spinner instead of flashing the
            sign-in button. */}
        {!isLoaded && (
          <p className="mt-6 text-xs text-ink-soft">Loading…</p>
        )}
      </motion.div>
    </main>
  );
}
