'use client';

import Link from 'next/link';
import { useUser } from '@clerk/nextjs';
import { FileStack, MessageSquare, RefreshCcw } from 'lucide-react';

import { useFiles, useOnboarding, useSessions } from '@/lib/queries';

export const runtime = 'edge';

export default function SettingsPage() {
  const { user } = useUser();
  const filesQuery = useFiles();
  const sessionsQuery = useSessions();
  const onboarding = useOnboarding();

  const files = filesQuery.data ?? [];
  const sessions = sessionsQuery.data ?? [];
  const totalChunks = files.reduce((sum, f) => sum + (f.chunk_count || 0), 0);

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <h1 className="font-serif text-2xl text-ink">Settings</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Your account, library, and conversation snapshot — sourced from D1 in
        real time.
      </p>

      <section className="mt-6 rounded-2xl border border-parchment-warm bg-parchment p-6">
        <h2 className="font-serif text-lg text-ink">Account</h2>
        <dl className="mt-3 grid grid-cols-1 gap-y-2 text-sm sm:grid-cols-[140px_1fr]">
          <dt className="text-ink-muted">Name</dt>
          <dd className="text-ink">{user?.fullName ?? '—'}</dd>
          <dt className="text-ink-muted">Email</dt>
          <dd className="text-ink">
            {user?.primaryEmailAddress?.emailAddress ?? '—'}
          </dd>
          <dt className="text-ink-muted">Clerk id</dt>
          <dd className="font-mono text-xs text-ink-soft">{user?.id ?? '—'}</dd>
        </dl>
      </section>

      <section className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard label="Files" value={files.length} href="/library" Icon={FileStack} />
        <StatCard
          label="Indexed chunks"
          value={totalChunks}
          href="/library"
          Icon={RefreshCcw}
        />
        <StatCard
          label="Conversations"
          value={sessions.length}
          href="/chat"
          Icon={MessageSquare}
        />
      </section>

      <section className="mt-6 rounded-2xl border border-parchment-warm bg-parchment p-6">
        <h2 className="font-serif text-lg text-ink">Onboarding</h2>
        <p className="mt-1 text-sm text-ink-muted">
          {onboarding.isComplete
            ? 'All set — every feature is unlocked.'
            : `Currently on step ${
                onboarding.step === 'done' ? 'final' : onboarding.step
              } of 3.`}
        </p>
        <ul className="mt-4 space-y-1 text-sm">
          <li className="flex items-center gap-2">
            <span className={onboarding.readyFilesCount > 0 ? 'text-accent' : 'text-ink-muted'}>
              {onboarding.readyFilesCount > 0 ? '✓' : '○'}
            </span>
            Step 1 — Upload your first file
          </li>
          <li className="flex items-center gap-2">
            <span className={onboarding.sessionsCount > 0 ? 'text-accent' : 'text-ink-muted'}>
              {onboarding.sessionsCount > 0 ? '✓' : '○'}
            </span>
            Step 2 — Ask your first question
          </li>
          <li className="flex items-center gap-2">
            <span className={onboarding.isComplete ? 'text-accent' : 'text-ink-muted'}>
              {onboarding.isComplete ? '✓' : '○'}
            </span>
            Step 3 — Generate a practice exam
          </li>
        </ul>
      </section>

      <section className="mt-6 rounded-2xl border border-parchment-warm bg-parchment p-6">
        <h2 className="font-serif text-lg text-ink">About</h2>
        <p className="mt-2 text-sm text-ink-muted">
          LawGPT is a personal legal-study assistant. All retrieval is
          namespace-isolated to your account — no cross-tenant reads.
        </p>
        <p className="mt-2 text-xs text-ink-soft">
          Vector backend: pgvector · Blob: R2 · Auth: Clerk
        </p>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  href,
  Icon,
}: {
  label: string;
  value: number;
  href: '/library' | '/chat' | '/exam';
  Icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <Link
      href={href}
      className="flex items-center gap-3 rounded-2xl border border-parchment-warm bg-parchment p-4 transition hover:bg-parchment-warm/40"
    >
      <Icon className="h-5 w-5 text-accent" />
      <div>
        <p className="text-xs uppercase tracking-wide text-ink-soft">{label}</p>
        <p className="font-serif text-xl text-ink">{value}</p>
      </div>
    </Link>
  );
}
