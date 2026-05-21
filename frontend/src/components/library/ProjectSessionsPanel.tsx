'use client';

import React from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { ArrowUpRight, MessageSquare, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/Button';
import { SkeletonList } from '@/components/ui/Skeleton';
import { useSessions } from '@/lib/queries';
import type { SessionSummary } from '@/lib/types';

function formatSessionDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value));
}

export function ProjectSessionsPanel({ projectId }: { projectId: string }) {
  const sessionsQuery = useSessions({ projectId });

  return (
    <ProjectSessionsPanelContent
      projectId={projectId}
      sessions={sessionsQuery.data ?? []}
      isLoading={sessionsQuery.isLoading}
    />
  );
}

export function ProjectSessionsPanelContent({
  projectId,
  sessions,
  isLoading,
}: {
  projectId: string;
  sessions: SessionSummary[];
  isLoading: boolean;
}) {
  const chatHref = `/chat?project=${projectId}` as Route;
  const visibleSessions = sessions.slice(0, 5);

  return (
    <section className="mb-8 border-b border-parchment-warm pb-7">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-accent" />
          <div>
            <h2 className="font-serif text-xl text-ink">Recent chats</h2>
            <p className="text-sm text-ink-muted">Resume conversations tied to this subject.</p>
          </div>
        </div>
        <Button asChild size="sm">
          <Link href={chatHref}>
            <MessageSquare className="h-4 w-4" />
            New subject chat
          </Link>
        </Button>
      </div>

      {isLoading ? (
        <SkeletonList rows={3} className="mt-4" />
      ) : visibleSessions.length === 0 ? (
        <div className="mt-4 rounded-lg border border-dashed border-parchment-warm bg-parchment px-4 py-5">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 rounded-md bg-parchment-warm p-2 text-accent">
              <Sparkles className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <p className="font-medium text-ink">No subject chats yet</p>
              <p className="mt-1 text-sm leading-5 text-ink-muted">
                Start a scoped chat here and it will stay attached to this workspace for later.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <ul className="mt-4 divide-y divide-parchment-warm overflow-hidden rounded-lg border border-parchment-warm bg-parchment">
          {visibleSessions.map((session) => {
            const href = `/chat/${session.id}` as Route;
            return (
              <li key={session.id}>
                <Link
                  href={href}
                  className="group flex items-center justify-between gap-4 px-4 py-3 transition hover:bg-parchment-warm"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-ink">
                      {session.title || 'Untitled chat'}
                    </span>
                    <span className="mt-1 block text-xs text-ink-soft">
                      Updated {formatSessionDate(session.updated_at)}
                    </span>
                  </span>
                  <ArrowUpRight className="h-4 w-4 shrink-0 text-ink-soft transition group-hover:text-accent" />
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
