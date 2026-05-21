'use client';

import { useEffect, useRef, useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import {
  QueryClient,
  QueryClientProvider,
  hydrate,
  type DehydratedState,
} from '@tanstack/react-query';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';
import {
  persistQueryClientSubscribe,
  type PersistedClient,
} from '@tanstack/react-query-persist-client';

type SyncStoragePersister = ReturnType<typeof createSyncStoragePersister>;

import { setTokenProvider } from '@/lib/api';

/** Bumped whenever cached query shapes change in a backwards-incompatible
 *  way; mismatched values invalidate the entire persisted cache on next load. */
const CACHE_BUSTER = 'v1';
const CACHE_MAX_AGE_MS = 1000 * 60 * 60 * 24 * 7; // 7 days
const CACHE_KEY = 'ashgpt-rq-cache';

const PERSISTED_QUERY_KEYS = new Set(['sessions', 'messages', 'files', 'users']);

/** Read the persisted snapshot synchronously inside the QueryClient's
 *  lazy initializer. Doing this *before* render — instead of via
 *  ``PersistQueryClientProvider``'s async ``restoreClient`` — means the
 *  first paint already has hydrated data, eliminating the flicker where
 *  the sidebar's history list briefly empties between tab navigations. */
function buildClient(): { client: QueryClient; persister: SyncStoragePersister | null } {
  const client = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        refetchOnWindowFocus: false,
        gcTime: CACHE_MAX_AGE_MS,
      },
    },
  });
  if (typeof window === 'undefined') return { client, persister: null };

  const persister = createSyncStoragePersister({
    storage: window.localStorage,
    key: CACHE_KEY,
  });

  // Manual sync restore from localStorage. ``createSyncStoragePersister``
  // exposes restoreClient as async but the underlying IO is sync — we
  // skip the Promise wrapper so hydration happens before React commits.
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (raw) {
      const persisted = JSON.parse(raw) as PersistedClient;
      const fresh =
        persisted.timestamp &&
        Date.now() - persisted.timestamp < CACHE_MAX_AGE_MS;
      const compatible = persisted.buster === CACHE_BUSTER;
      if (fresh && compatible && persisted.clientState) {
        hydrate(client, persisted.clientState as unknown as DehydratedState);
      } else {
        window.localStorage.removeItem(CACHE_KEY);
      }
    }
  } catch {
    // Corrupted cache — drop it and continue with a cold client.
    try {
      window.localStorage.removeItem(CACHE_KEY);
    } catch {
      /* ignore */
    }
  }

  return { client, persister };
}

/** Client-side providers mounted by the (app) layout.
 *
 *  - QueryClient is created once per mount with hydrated cache (sync).
 *  - Cache writes are streamed back to localStorage via
 *    ``persistQueryClientSubscribe`` so any update made during the
 *    session is persisted without re-entering an async hydration path.
 *  - The Clerk ``getToken`` callback is registered with the API layer so
 *    ``request()`` can replay a 401 with a freshly-minted JWT exactly once.
 *  - Persisted cache is cleared on sign-out so a different user logging in
 *    on the same browser never sees stale data from the previous account.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [{ client, persister }] = useState(buildClient);

  // Stream cache mutations back to localStorage. Returns an unsubscribe
  // that we keep around for the unmount cleanup.
  useEffect(() => {
    if (!persister) return;
    const unsubscribe = persistQueryClientSubscribe({
      queryClient: client,
      persister,
      buster: CACHE_BUSTER,
      dehydrateOptions: {
        shouldDehydrateQuery: (query) => {
          const [key] = query.queryKey as [string, ...unknown[]];
          return PERSISTED_QUERY_KEYS.has(key);
        },
      },
    });
    return () => unsubscribe();
  }, [client, persister]);

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

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
