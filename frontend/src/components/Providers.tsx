'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { QueryClient } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';

import { setTokenProvider } from '@/lib/api';

/** Bumped whenever the cached query shapes change in a backwards-incompatible
 *  way; mismatched values invalidate the entire persisted cache on next load. */
const CACHE_BUSTER = 'v1';
const CACHE_MAX_AGE_MS = 1000 * 60 * 60 * 24 * 7; // 7 days

/** Client-side providers mounted by the (app) layout.
 *
 *  - QueryClient is created once per mount.
 *  - Cached queries (sessions, messages, files, current user) are persisted
 *    to localStorage so a hard reload paints from cache before the network
 *    round-trip — no "loading conversation…" flash.
 *  - The Clerk ``getToken`` callback is registered with the API layer so
 *    ``request()`` can replay a 401 with a freshly-minted JWT exactly once.
 *  - Persisted cache is cleared on sign-out so a different user logging in
 *    on the same browser never sees stale data from the previous account.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
            // Cached data is persistable for up to a week; gcTime governs
            // how long it survives in-memory after the last subscriber
            // unmounts and must exceed staleTime by a clear margin so
            // persisted snapshots are written before eviction.
            gcTime: CACHE_MAX_AGE_MS,
          },
        },
      }),
  );

  const persister = useMemo(
    () =>
      typeof window === 'undefined'
        ? undefined
        : createSyncStoragePersister({
            storage: window.localStorage,
            key: 'ashgpt-rq-cache',
          }),
    [],
  );

  // Wire Clerk's getToken into the API module so 401 replays can refresh the
  // JWT without dragging React hooks into every queryFn.
  const { getToken, isSignedIn, isLoaded } = useAuth();
  useEffect(() => {
    setTokenProvider(async (opts) => {
      try {
        return await getToken(opts);
      } catch {
        return null;
      }
    });
    return () => setTokenProvider(null);
  }, [getToken]);

  // Bust the persisted cache when the user signs out so a different user
  // on the same device never reads the previous user's sessions/messages.
  const wasSignedIn = useRef(false);
  useEffect(() => {
    if (!isLoaded) return;
    if (isSignedIn) {
      wasSignedIn.current = true;
      return;
    }
    if (wasSignedIn.current && persister) {
      void persister.removeClient();
      client.clear();
      wasSignedIn.current = false;
    }
  }, [isLoaded, isSignedIn, persister, client]);

  if (!persister) {
    // SSR path: render children without persistence, hydration will swap
    // in the persisted version on the client.
    return <>{children}</>;
  }

  return (
    <PersistQueryClientProvider
      client={client}
      persistOptions={{
        persister,
        maxAge: CACHE_MAX_AGE_MS,
        buster: CACHE_BUSTER,
        dehydrateOptions: {
          shouldDehydrateQuery: (query) => {
            const [key] = query.queryKey as [string, ...unknown[]];
            return (
              key === 'sessions' ||
              key === 'messages' ||
              key === 'files' ||
              key === 'users'
            );
          },
        },
      }}
    >
      {children}
    </PersistQueryClientProvider>
  );
}
