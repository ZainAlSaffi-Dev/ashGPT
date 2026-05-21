'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FileStack, Trash2 } from 'lucide-react';
import { useAuth } from '@clerk/nextjs';

import { Button } from '@/components/ui/Button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/Dialog';
import { SkeletonList } from '@/components/ui/Skeleton';
import { deleteFile } from '@/lib/api';
import { useFiles, useInvalidateFiles } from '@/lib/queries';
import type { FileListScope, FileMeta } from '@/lib/types';
import { cn } from '@/lib/utils';

function fileSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function FileStatus({ file }: { file: FileMeta }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium capitalize',
        file.status === 'ready' && 'bg-accent/10 text-accent',
        (file.status === 'uploaded' || file.status === 'processing' || file.status === 'queued') &&
          'bg-ink/5 text-ink-muted',
        file.status === 'failed' && 'bg-red-50 text-red-700',
      )}
    >
      {file.status}
    </span>
  );
}

export function LibraryFileList({
  scope = {},
  targetFile,
  emptyTitle = 'No files yet',
  emptyDescription = 'Drop a PDF, DOCX, image, or markdown file above to get started.',
}: {
  scope?: FileListScope;
  targetFile?: string | null;
  emptyTitle?: string;
  emptyDescription?: string;
}) {
  const { getToken } = useAuth();
  const filesQuery = useFiles(scope);
  const invalidateFiles = useInvalidateFiles(scope);
  const files = filesQuery.data ?? [];
  const listRef = useRef<HTMLUListElement | null>(null);
  const didScrollRef = useRef<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<FileMeta | null>(null);
  const [deleting, setDeleting] = useState(false);

  const matchId = useMemo(() => {
    if (!targetFile || !files.length) return null;
    const byId = files.find((f) => f.id === targetFile);
    if (byId) return byId.id;
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
    <>
      {targetFile && !matchId && !filesQuery.isLoading && (
        <p className="mt-3 text-xs text-ink-soft">
          Could not find <span className="font-medium text-ink">{targetFile}</span> in this library.
        </p>
      )}
      {filesQuery.isLoading ? (
        <SkeletonList rows={4} className="mt-3" />
      ) : files.length === 0 ? (
        <EmptyFiles title={emptyTitle} description={emptyDescription} />
      ) : (
        <ul
          ref={listRef}
          className="mt-3 overflow-hidden rounded-lg border border-parchment-warm bg-parchment"
        >
          <AnimatePresence initial={false}>
            {files.map((file) => {
              const isHit = matchId === file.id;
              return (
                <motion.li
                  key={file.id}
                  layout
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: 12, height: 0, padding: 0 }}
                  transition={{ duration: 0.18 }}
                  id={`file-${file.id}`}
                  data-file-slug={fileSlug(file.name)}
                  className={cn(
                    'group flex scroll-mt-20 items-center justify-between gap-4 border-b border-parchment-warm px-4 py-3 last:border-b-0',
                    isHit && 'bg-accent/10 ring-1 ring-inset ring-accent',
                  )}
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white text-accent ring-1 ring-parchment-warm">
                      <FileStack className="h-4 w-4" />
                    </span>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-ink">{file.name}</p>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-ink-soft">
                        <span>{file.doc_type}</span>
                        {file.week && <span>{file.week}</span>}
                        <span>
                          {file.chunk_count} chunk{file.chunk_count === 1 ? '' : 's'}
                        </span>
                        <FileStatus file={file} />
                      </div>
                      {file.error && <p className="mt-1 text-xs text-red-600">{file.error}</p>}
                    </div>
                  </div>
                  <button
                    onClick={() => setPendingDelete(file)}
                    className="rounded-md p-2 text-ink-soft opacity-100 transition hover:bg-red-50 hover:text-red-600 md:opacity-0 md:group-hover:opacity-100"
                    aria-label={`Delete ${file.name}`}
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
        onOpenChange={(open) => !open && setPendingDelete(null)}
      >
        <DialogContent open={!!pendingDelete}>
          <DialogTitle className="font-serif text-lg text-ink">Delete file?</DialogTitle>
          <DialogDescription className="mt-2 text-sm text-ink-muted">
            "{pendingDelete?.name}" and all its chunks will be removed. This can't be undone.
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
              {deleting ? 'Deleting...' : 'Delete'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

function EmptyFiles({ title, description }: { title: string; description: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      className="mt-3 rounded-lg border border-dashed border-parchment-warm bg-parchment px-6 py-9 text-center"
    >
      <FileStack className="mx-auto h-8 w-8 text-accent" />
      <p className="mt-3 font-serif text-lg text-ink">{title}</p>
      <p className="mx-auto mt-1 max-w-sm text-sm text-ink-muted">{description}</p>
    </motion.div>
  );
}
