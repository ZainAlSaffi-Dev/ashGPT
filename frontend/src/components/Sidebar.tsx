'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { AnimatePresence, motion } from 'framer-motion';
import type { Route } from 'next';
import {
  BookMarked,
  BookOpen,
  ChevronDown,
  FileStack,
  GraduationCap,
  MessageSquare,
  Plus,
  Settings,
  Sparkles,
  Trash2,
} from 'lucide-react';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/Dialog';
import { Button } from '@/components/ui/Button';
import { SkeletonList } from '@/components/ui/Skeleton';
import { useDeleteSession, useProjects, useSessions } from '@/lib/queries';
import type { Project, SessionSummary } from '@/lib/types';
import { cn } from '@/lib/utils';

const nav = [
  { href: '/chat', label: 'Chat', Icon: MessageSquare },
  { href: '/library', label: 'Library', Icon: FileStack },
  { href: '/exam', label: 'Exam', Icon: GraduationCap },
  { href: '/settings', label: 'Settings', Icon: Settings },
] as const;

const subjectPalette = ['#7a3b2e', '#315f72', '#596b3a', '#6f4d7a', '#8a6536'];

function subjectColor(project: Project, index: number) {
  return project.color || subjectPalette[index % subjectPalette.length];
}

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const sessionsQuery = useSessions();
  const projectsQuery = useProjects();
  const deleteSession = useDeleteSession();

  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const [subjectsOpen, setSubjectsOpen] = useState(true);
  const projects = projectsQuery.data ?? [];
  const globalSessions = (sessionsQuery.data ?? []).filter((session) => !session.project_id);

  const confirmDelete = () => {
    if (!pendingDelete) return;
    const id = pendingDelete.id;
    setPendingDelete(null);
    deleteSession.mutate(id, {
      onSettled: () => {
        if (pathname === `/chat/${id}`) router.push('/chat');
      },
    });
  };

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-parchment-warm bg-parchment p-4">
      <Link
        href="/"
        className="mb-6 flex items-center gap-2 font-serif text-xl text-ink transition hover:opacity-90"
      >
        <motion.span
          initial={{ rotate: -8, scale: 0.9 }}
          animate={{ rotate: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 240, damping: 16 }}
          className="inline-flex"
        >
          <BookMarked className="h-5 w-5 text-accent" />
        </motion.span>
        ashGPT
      </Link>

      <nav className="flex flex-col gap-1">
        {nav.map(({ href, label, Icon }) => {
          const active = pathname === href || pathname.startsWith(href + '/');
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                'group relative flex items-center gap-3 rounded-md px-3 py-2 text-sm transition',
                active
                  ? 'text-ink'
                  : 'text-ink-muted hover:text-ink',
              )}
            >
              {active && (
                <motion.span
                  layoutId="sidebar-active-pill"
                  className="absolute inset-0 -z-0 rounded-md bg-parchment-warm"
                  transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                />
              )}
              <span className="relative z-10 inline-flex items-center gap-3">
                <Icon className="h-4 w-4" />
                {label}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-4">
        <button
          type="button"
          onClick={() => setSubjectsOpen((open) => !open)}
          className="flex w-full items-center justify-between rounded-md px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-soft transition hover:bg-parchment-warm hover:text-ink"
          aria-expanded={subjectsOpen}
        >
          <span className="inline-flex items-center gap-2">
            <BookOpen className="h-3.5 w-3.5" />
            Subjects
          </span>
          <ChevronDown
            className={cn(
              'h-3.5 w-3.5 transition-transform',
              subjectsOpen && 'rotate-180',
            )}
          />
        </button>
        <AnimatePresence initial={false}>
          {subjectsOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.16 }}
              className="overflow-hidden"
            >
              <SubjectLinks
                projects={projects}
                activePath={pathname}
                isLoading={projectsQuery.isLoading}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

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
          sessions={globalSessions}
          activePath={pathname}
          isLoading={sessionsQuery.isLoading}
          onDelete={(s) => setPendingDelete(s)}
        />
      </div>

      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent open={!!pendingDelete}>
          <DialogTitle className="font-serif text-lg text-ink">Delete chat?</DialogTitle>
          <DialogDescription className="mt-2 text-sm text-ink-muted">
            “{pendingDelete?.title || 'this chat'}” will be removed. This can't be undone.
          </DialogDescription>
          <div className="mt-6 flex justify-end gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPendingDelete(null)}
            >
              Cancel
            </Button>
            <Button variant="danger" size="sm" onClick={confirmDelete}>
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </aside>
  );
}

function SubjectLinks({
  projects,
  activePath,
  isLoading,
}: {
  projects: Project[];
  activePath: string;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <SkeletonList rows={3} className="mt-2" />;
  }
  if (!projects.length) {
    return (
      <Link
        href="/library"
        className="mt-2 flex items-center gap-2 rounded-md px-3 py-2 text-sm text-ink-muted transition hover:bg-parchment-warm hover:text-ink"
      >
        <Plus className="h-3.5 w-3.5" />
        Create subject
      </Link>
    );
  }
  return (
    <ul className="mt-1 flex max-h-44 flex-col gap-0.5 overflow-y-auto pr-1">
      {projects.map((project, index) => {
        const href = `/library/${project.id}` as Route;
        const active = activePath === href;
        return (
          <li key={project.id}>
            <Link
              href={href}
              className={cn(
                'flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition',
                active
                  ? 'bg-parchment-warm text-ink'
                  : 'text-ink-muted hover:bg-parchment-warm hover:text-ink',
              )}
              title={project.name}
            >
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: subjectColor(project, index) }}
              />
              <span className="truncate">{project.name}</span>
            </Link>
          </li>
        );
      })}
    </ul>
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
  onDelete: (s: SessionSummary) => void;
}) {
  if (isLoading) {
    return <SkeletonList rows={5} className="mt-1" />;
  }
  if (!sessions.length) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="mt-3 rounded-lg border border-dashed border-parchment-warm px-3 py-4 text-center"
      >
        <Sparkles className="mx-auto h-4 w-4 text-accent" />
        <p className="mt-2 text-xs text-ink-soft">
          No conversations yet. Ask your first question to start.
        </p>
      </motion.div>
    );
  }
  return (
    <ul className="flex flex-col gap-0.5">
      <AnimatePresence initial={false}>
        {sessions.map((s) => {
          const href = `/chat/${s.id}` as Route;
          const active = activePath === href;
          return (
            <motion.li
              key={s.id}
              layout
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8, height: 0, marginTop: 0, marginBottom: 0 }}
              transition={{ duration: 0.18 }}
              className="group relative"
            >
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
                  onDelete(s);
                }}
                aria-label={`Delete ${s.title || 'chat'}`}
                className="absolute right-1.5 top-1/2 hidden -translate-y-1/2 rounded p-1 text-ink-soft transition hover:bg-parchment hover:text-red-600 group-hover:block"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </motion.li>
          );
        })}
      </AnimatePresence>
    </ul>
  );
}
