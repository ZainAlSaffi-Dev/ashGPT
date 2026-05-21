'use client';

import { useAuth } from '@clerk/nextjs';
import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationOptions,
} from '@tanstack/react-query';

import {
  deleteSession,
  getMe,
  listFiles,
  listMessages,
  listSessions,
  markOnboarded,
  withAuth,
  type UserMe,
} from './api';
import type { SessionSummary } from './types';
import { useAuthReady } from './useAuthReady';

/** Files list, cached so library + exam + onboarding share one fetch.
 *  Polls every 4 s while any file is still in a non-terminal status
 *  (``uploaded`` / ``processing`` / ``queued``) so the user sees the
 *  transition to ``ready`` without manual refresh. */
interface QueryGateOptions {
  enabled?: boolean;
}

export function useFiles(options: QueryGateOptions = {}) {
  const { getToken } = useAuth();
  const authReady = useAuthReady();
  return useQuery({
    queryKey: ['files'],
    enabled: authReady && options.enabled !== false,
    queryFn: () => withAuth(getToken, (token) => listFiles(token)),
    // Fallback so an in-flight file that never reaches a terminal status
    // still gets revalidated on revisit, instead of polling forever.
    staleTime: 5 * 60_000,
    // Tab nav must feel instant — keep cached files on screen and
    // revalidate in the background instead of flashing the skeleton.
    refetchOnMount: false,
    placeholderData: keepPreviousData,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data || data.length === 0) return false;
      const inFlight = data.some(
        (f) => f.status === 'uploaded' || f.status === 'processing' || f.status === 'queued',
      );
      return inFlight ? 4_000 : false;
    },
  });
}

export function useSessions(options: QueryGateOptions = {}) {
  const { getToken } = useAuth();
  const authReady = useAuthReady();
  return useQuery({
    queryKey: ['sessions'],
    enabled: authReady && options.enabled !== false,
    queryFn: () => withAuth(getToken, (token) => listSessions(token)),
    // Sidebar list shouldn't flicker on tab focus; it's append-only mostly
    // and gets invalidated on create/delete.
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    placeholderData: keepPreviousData,
  });
}

export function useMessages(sessionId: string | null | undefined) {
  const { getToken } = useAuth();
  const authReady = useAuthReady();
  return useQuery({
    queryKey: ['messages', sessionId],
    enabled: authReady && !!sessionId,
    queryFn: () => withAuth(getToken, (token) => listMessages(sessionId!, token)),
    // Persisted messages are immutable, so navigation back into a chat
    // shouldn't trigger a refetch on focus. New turns are pushed into
    // local useChat state and invalidated by ChatSurface when the
    // assistant message is committed server-side.
    staleTime: 5 * 60_000,
    gcTime: 10 * 60_000,
    // Reopening an old chat paints from that session's own cache. Do not use
    // keepPreviousData here: during rapid session switches it can briefly show
    // the previous conversation under the new URL.
    refetchOnMount: false,
  });
}

/** Used by upload + delete to invalidate the shared file list cache. */
export function useInvalidateFiles() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ['files'] });
}

/** Optimistic session delete: removes from the cached list immediately,
 *  rolls back if the API call fails. Sidebar uses this so the row
 *  disappears the moment the user confirms — no spinner lag. */
export function useDeleteSession(
  options?: Pick<UseMutationOptions<void, Error, string, { previous?: SessionSummary[] }>, 'onSuccess'>,
) {
  const { getToken } = useAuth();
  const qc = useQueryClient();
  return useMutation<void, Error, string, { previous?: SessionSummary[] }>({
    mutationFn: (id: string) => withAuth(getToken, (token) => deleteSession(id, token)),
    onMutate: async (id) => {
      await qc.cancelQueries({ queryKey: ['sessions'] });
      const previous = qc.getQueryData<SessionSummary[]>(['sessions']);
      if (previous) {
        qc.setQueryData<SessionSummary[]>(
          ['sessions'],
          previous.filter((s) => s.id !== id),
        );
      }
      return { previous };
    },
    onError: (_err, _id, ctx) => {
      if (ctx?.previous) qc.setQueryData(['sessions'], ctx.previous);
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['sessions'] });
    },
    ...options,
  });
}

/** Current user — drives the welcome tour gate.
 *  ``onboarded_at`` flips from null → ISO timestamp the first time the
 *  user finishes (or skips) the tour. */
export function useCurrentUser() {
  const { getToken } = useAuth();
  const authReady = useAuthReady();
  return useQuery<UserMe>({
    queryKey: ['users', 'me'],
    enabled: authReady,
    queryFn: () => withAuth(getToken, (token) => getMe(token)),
    staleTime: 5 * 60_000,
    refetchOnMount: false,
    placeholderData: keepPreviousData,
  });
}

export function useMarkOnboarded() {
  const { getToken } = useAuth();
  const qc = useQueryClient();
  return useMutation<UserMe, Error, void>({
    mutationFn: () => withAuth(getToken, (token) => markOnboarded(token)),
    onSuccess: (user) => {
      qc.setQueryData<UserMe>(['users', 'me'], user);
    },
  });
}

/** Returns the user's progress through the welcome funnel.
 *  Steps:
 *    1. Upload first file
 *    2. Ask first question (creates first session)
 *    3. Generate first exam — implicit once steps 1 + 2 done.
 */
export interface OnboardingState {
  filesCount: number;
  readyFilesCount: number;
  sessionsCount: number;
  step: 1 | 2 | 3 | 'done';
  isComplete: boolean;
  isLoading: boolean;
}

export function useOnboarding(options: QueryGateOptions = {}): OnboardingState {
  const enabled = options.enabled !== false;
  const filesQuery = useFiles({ enabled });
  const sessionsQuery = useSessions({ enabled });

  const filesCount = filesQuery.data?.length ?? 0;
  const readyFilesCount =
    filesQuery.data?.filter((f) => f.status === 'ready').length ?? 0;
  const sessionsCount = sessionsQuery.data?.length ?? 0;

  let step: OnboardingState['step'] = 'done';
  if (readyFilesCount === 0) step = 1;
  else if (sessionsCount === 0) step = 2;
  else step = 'done';

  return {
    filesCount,
    readyFilesCount,
    sessionsCount,
    step,
    isComplete: step === 'done',
    isLoading: enabled && (filesQuery.isLoading || sessionsQuery.isLoading),
  };
}
