'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FileStack, Trash2 } from 'lucide-react';
import { useAuth } from '@clerk/nextjs';
import { useSearchParams } from 'next/navigation';

import { Dropzone } from '@/components/Dropzone';
import { OnboardingChecklist } from '@/components/OnboardingChecklist';
import { Button } from '@/components/ui/Button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/Dialog';
import { SkeletonList } from '@/components/ui/Skeleton';
import { deleteFile } from '@/lib/api';
import { useFiles, useInvalidateFiles, useOnboarding } from '@/lib/queries';
import type { FileMeta } from '@/lib/types';
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
  const didScrollRef = useRef<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<FileMeta | null>(null);
  const [deleting, setDeleting] = useState(false);

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
    requestAnimationFrame(() => {
      document
        .getElementById(`file-${matchId}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, [matchId]);

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      const token = (await getToken()) ?? undefined;
      await deleteFile(pendingDelete.id, token);
      await invalidateFiles();
    } finally {
      setDeleting(false);
      setPendingDelete(null);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <h1 className="font-serif text-2xl text-ink">Library</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Drop your readings, lecture slides, and notes. They&apos;ll be chunked,
          embedded, and made searchable for chat + exam generation.
        </p>
      </motion.div>

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
        <SkeletonList rows={3} className="mt-3" />
      ) : files.length === 0 ? (
        <EmptyFiles />
      ) : (
        <ul
          ref={listRef}
          className="mt-3 divide-y divide-parchment-warm rounded-lg border border-parchment-warm bg-parchment"
        >
          <AnimatePresence initial={false}>
            {files.map((f) => {
              const isHit = matchId === f.id;
              return (
                <motion.li
                  key={f.id}
                  layout
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: 12, height: 0, padding: 0 }}
                  transition={{ duration: 0.2 }}
                  id={`file-${f.id}`}
                  data-file-slug={fileSlug(f.name)}
                  className={cn(
                    'flex scroll-mt-20 items-center justify-between px-4 py-3',
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
                    onClick={() => setPendingDelete(f)}
                    className="text-ink-muted transition hover:text-red-600"
                    aria-label={`Delete ${f.name}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ul>
      )}

      <Dialog
        open={!!pendingDelete}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent open={!!pendingDelete}>
          <DialogTitle className="font-serif text-lg text-ink">Delete file?</DialogTitle>
          <DialogDescription className="mt-2 text-sm text-ink-muted">
            “{pendingDelete?.name}” and all its chunks will be removed. This can&apos;t be undone.
          </DialogDescription>
          <div className="mt-6 flex justify-end gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPendingDelete(null)}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={confirmDelete}
              disabled={deleting}
            >
              {deleting ? 'Deleting…' : 'Delete'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function EmptyFiles() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="mt-3 rounded-xl border border-dashed border-parchment-warm bg-parchment px-6 py-10 text-center"
    >
      <motion.div
        animate={{ y: [0, -4, 0] }}
        transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
        className="mx-auto inline-flex"
      >
        <FileStack className="h-8 w-8 text-accent" />
      </motion.div>
      <p className="mt-3 font-serif text-lg text-ink">No files yet</p>
      <p className="mx-auto mt-1 max-w-sm text-sm text-ink-muted">
        Drop a PDF, DOCX, image, or markdown file above to get started. Larger
        readings chunk in the background — you can keep using the rest of the
        app while they process.
      </p>
    </motion.div>
  );
}
