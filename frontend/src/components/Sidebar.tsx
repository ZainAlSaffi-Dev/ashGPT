'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@clerk/nextjs';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Route } from 'next';
import {
  BookMarked,
  FileStack,
  GraduationCap,
  MessageSquare,
  Plus,
  Settings,
  Trash2,
} from 'lucide-react';

import { deleteSession, listSessions } from '@/lib/api';
import type { SessionSummary } from '@/lib/types';
import { cn } from '@/lib/utils';

const nav = [
  { href: '/chat', label: 'Chat', Icon: MessageSquare },
  { href: '/library', label: 'Library', Icon: FileStack },
  { href: '/exam', label: 'Exam', Icon: GraduationCap },
  { href: '/settings', label: 'Settings', Icon: Settings },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { getToken, isSignedIn } = useAuth();

  const sessionsQuery = useQuery({
    queryKey: ['sessions'],
    enabled: !!isSignedIn,
    queryFn: async () => {
      const token = (await getToken()) ?? undefined;
      return listSessions(token);
    },
  });

  const onDeleteSession = async (id: string, title: string) => {
    if (!confirm(`Delete "${title || 'this chat'}"? This can't be undone.`)) return;
    const token = (await getToken()) ?? undefined;
    await deleteSession(id, token);
    await queryClient.invalidateQueries({ queryKey: ['sessions'] });
    // If the deleted session is the one currently routed to, bounce back to
    // the landing route so we don't render against a 404 session.
    if (pathname === `/chat/${id}`) {
      router.push('/chat');
    }
  };

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-parchment-warm bg-parchment p-4">
      <Link href="/" className="mb-6 flex items-center gap-2 font-serif text-xl text-ink">
        <BookMarked className="h-5 w-5 text-accent" />
        LawGPT
      </Link>

      <nav className="flex flex-col gap-1">
        {nav.map(({ href, label, Icon }) => {
          const active = pathname === href || pathname.startsWith(href + '/');
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition',
                active
                  ? 'bg-parchment-warm text-ink'
                  : 'text-ink-muted hover:bg-parchment-warm hover:text-ink',
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-6 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-soft">
          History
        </span>
        <Link
          href="/chat"
          className="rounded-md p-1 text-ink-muted transition hover:bg-parchment-warm hover:text-accent"
          aria-label="New chat"
        >
          <Plus className="h-3.5 w-3.5" />
        </Link>
      </div>

      <div className="mt-2 flex-1 overflow-y-auto">
        <SessionList
          sessions={sessionsQuery.data ?? []}
          activePath={pathname}
          isLoading={sessionsQuery.isLoading}
          onDelete={onDeleteSession}
        />
      </div>
    </aside>
  );
}

function SessionList({
  sessions,
  activePath,
  isLoading,
  onDelete,
}: {
  sessions: SessionSummary[];
  activePath: string;
  isLoading: boolean;
  onDelete: (id: string, title: string) => void | Promise<void>;
}) {
  if (isLoading) {
    return <div className="px-3 py-2 text-xs text-ink-soft">Loading…</div>;
  }
  if (!sessions.length) {
    return (
      <div className="px-3 py-2 text-xs text-ink-soft">No conversations yet.</div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5">
      {sessions.map((s) => {
        const href = `/chat/${s.id}` as Route;
        const active = activePath === href;
        return (
          <li key={s.id} className="group relative">
            <Link
              href={href}
              className={cn(
                'block truncate rounded-md px-3 py-1.5 pr-8 text-sm transition',
                active
                  ? 'bg-parchment-warm text-ink'
                  : 'text-ink-muted hover:bg-parchment-warm hover:text-ink',
              )}
              title={s.title}
            >
              {s.title || 'Untitled chat'}
            </Link>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                void onDelete(s.id, s.title);
              }}
              aria-label={`Delete ${s.title || 'chat'}`}
              className="absolute right-1.5 top-1/2 hidden -translate-y-1/2 rounded p-1 text-ink-soft transition hover:bg-parchment hover:text-red-600 group-hover:block"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
