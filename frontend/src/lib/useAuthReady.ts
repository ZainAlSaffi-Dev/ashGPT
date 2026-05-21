'use client';

import { useAuth } from '@clerk/nextjs';
import { useEffect, useRef, useState } from 'react';

/**
 * Returns true only once Clerk has loaded AND the in-memory session token
 * has been resolved at least once. The bug we're guarding against:
 *
 *   Right after sign-in/sign-up Clerk redirects with ``isSignedIn === true``
 *   while the in-memory JWT is still null. ``useQuery`` fires before
 *   ``getToken()`` resolves; the request goes out without an Authorization
 *   header; the edge worker returns 401; the UI flashes "server side error"
 *   until the user hard-refreshes.
 *
 * Gating ``enabled: useAuthReady()`` on every authed query removes the race:
 * no query fires before there's a real token to attach.
 *
 * In dev (NEXT_PUBLIC_DEV_USER set, no Clerk key), we short-circuit to
 * ``true`` immediately so local development isn't blocked.
 */
export function useAuthReady(): boolean {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  const [ready, setReady] = useState(false);
  const resolved = useRef(false);

  useEffect(() => {
    // Dev bypass — no Clerk in this build, every fetch uses X-Dev-User.
    if (typeof window !== 'undefined' && process.env.NEXT_PUBLIC_DEV_USER) {
      setReady(true);
      return;
    }
    if (!isLoaded) return;
    if (!isSignedIn) {
      setReady(false);
      resolved.current = false;
      return;
    }
    if (resolved.current) {
      setReady(true);
      return;
    }
    let cancelled = false;
    void (async () => {
      // Poll up to ~3s for the first non-null token. Each getToken() awaits
      // Clerk's internal state machine; we don't trust a single null read.
      for (let i = 0; i < 12 && !cancelled; i++) {
        const t = await getToken().catch(() => null);
        if (t) {
          resolved.current = true;
          setReady(true);
          return;
        }
        await new Promise((r) => setTimeout(r, 250));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isLoaded, isSignedIn, getToken]);

  return ready;
}
