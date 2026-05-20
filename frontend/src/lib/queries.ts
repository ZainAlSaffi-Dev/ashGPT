'use client';

import { useAuth } from '@clerk/nextjs';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import { listFiles, listMessages, listSessions } from './api';

/** Files list, cached so library + exam + onboarding share one fetch.
 *  Polls every 4 s while any file is still in a non-terminal status
 *  (``uploaded`` / ``processing`` / ``queued``) so the user sees the
 *  transition to ``ready`` without manual refresh. */
export function useFiles() {
  const { getToken, isSignedIn } = useAuth();
  return useQuery({
    queryKey: ['files'],
    enabled: !!isSignedIn,
    queryFn: async () => {
      const token = (await getToken()) ?? undefined;
      return listFiles(token);
    },
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data || data.length === 0) return false;
      const inFlight = data.some((f) =>
        f.status === 'uploaded' || f.status === 'processing' || f.status === 'queued',
      );
      return inFlight ? 4_000 : false;
    },
  });
}

export function useSessions() {
  const { getToken, isSignedIn } = useAuth();
  return useQuery({
    queryKey: ['sessions'],
    enabled: !!isSignedIn,
    queryFn: async () => {
      const token = (await getToken()) ?? undefined;
      return listSessions(token);
    },
  });
}

export function useMessages(sessionId: string | null | undefined) {
  const { getToken, isSignedIn } = useAuth();
  return useQuery({
    queryKey: ['messages', sessionId],
    enabled: !!isSignedIn && !!sessionId,
    queryFn: async () => {
      const token = (await getToken()) ?? undefined;
      return listMessages(sessionId!, token);
    },
  });
}

/** Used by upload + delete to invalidate the shared file list cache. */
export function useInvalidateFiles() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ['files'] });
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

export function useOnboarding(): OnboardingState {
  const filesQuery = useFiles();
  const sessionsQuery = useSessions();

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
    isLoading: filesQuery.isLoading || sessionsQuery.isLoading,
  };
}
