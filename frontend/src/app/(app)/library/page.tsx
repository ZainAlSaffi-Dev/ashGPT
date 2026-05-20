'use client';

import { useEffect, useMemo, useRef } from 'react';
import { Trash2 } from 'lucide-react';
import { useAuth } from '@clerk/nextjs';
import { useSearchParams } from 'next/navigation';

import { Dropzone } from '@/components/Dropzone';
import { OnboardingChecklist } from '@/components/OnboardingChecklist';
import { deleteFile } from '@/lib/api';
import { useFiles, useInvalidateFiles, useOnboarding } from '@/lib/queries';
import { cn } from '@/lib/utils';

function fileSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export default function LibraryPage() {
  const { getToken } = useAuth();
  const filesQuery = useFiles();
  const invalidateFiles = useInvalidateFiles();
  const onboarding = useOnboarding();
  const searchParams = useSearchParams();
  const targetFile = searchParams.get('file');

  const files = filesQuery.data ?? [];
  const listRef = useRef<HTMLUListElement | null>(null);
  // Only scroll once per (file param, file load) so the user can scroll
  // away after landing.
  const didScrollRef = useRef<string | null>(null);

  const matchId = useMemo(() => {
    if (!targetFile || !files.length) return null;
    const exact = files.find((f) => f.name === targetFile);
    if (exact) return exact.id;
    const ci = files.find((f) => f.name.toLowerCase() === targetFile.toLowerCase());
    return ci?.id ?? null;
  }, [files, targetFile]);

  useEffect(() => {
    if (!matchId || didScrollRef.current === matchId) return;
    didScrollRef.current = matchId;
    // Defer to the next frame so the list has mounted before we scroll.
    requestAnimationFrame(() => {
      document
        .getElementById(`file-${matchId}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, [matchId]);

  const onDelete = async (id: string) => {
    if (!confirm('Delete this file and all its chunks? This cannot be undone.')) return;
    const token = (await getToken()) ?? undefined;
    await deleteFile(id, token);
    await invalidateFiles();
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <h1 className="font-serif text-2xl text-ink">Library</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Drop your readings, lecture slides, and notes. They&apos;ll be chunked,
        embedded, and made searchable for chat + exam generation.
      </p>

      {!onboarding.isComplete && (
        <div className="mt-6">
          <OnboardingChecklist variant="full" />
        </div>
      )}

      <div className="mt-6">
        <Dropzone onComplete={() => void invalidateFiles()} />
      </div>

      <h2 className="mt-8 font-serif text-lg text-ink">Your files</h2>
      {targetFile && !matchId && !filesQuery.isLoading && (
        <p className="mt-3 text-xs text-ink-soft">
          Could not find <span className="font-medium text-ink">{targetFile}</span> in your library.
        </p>
      )}
      {filesQuery.isLoading ? (
        <p className="mt-3 text-sm text-ink-muted">Loading…</p>
      ) : files.length === 0 ? (
        <p className="mt-3 text-sm text-ink-muted">No files yet.</p>
      ) : (
        <ul
          ref={listRef}
          className="mt-3 divide-y divide-parchment-warm rounded-lg border border-parchment-warm bg-parchment"
        >
          {files.map((f) => {
            const isHit = matchId === f.id;
            return (
              <li
                key={f.id}
                id={`file-${f.id}`}
                data-file-slug={fileSlug(f.name)}
                className={cn(
                  'flex scroll-mt-20 items-center justify-between px-4 py-3 transition',
                  isHit && 'bg-accent/10 ring-1 ring-accent',
                )}
              >
                <div>
                  <p className="text-ink">{f.name}</p>
                  <p className="text-xs text-ink-soft">
                    {f.doc_type}
                    {f.week ? ` · ${f.week}` : ''}
                    {' · '}
                    {f.chunk_count} chunk{f.chunk_count === 1 ? '' : 's'}
                    {' · '}
                    <span
                      className={cn(
                        f.status === 'ready' && 'text-accent',
                        f.status === 'failed' && 'text-red-600',
                      )}
                    >
                      {f.status}
                    </span>
                  </p>
                  {f.error && <p className="text-xs text-red-600">{f.error}</p>}
                </div>
                <button
                  onClick={() => onDelete(f.id)}
                  className="text-ink-muted hover:text-red-600"
                  aria-label={`Delete ${f.name}`}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
