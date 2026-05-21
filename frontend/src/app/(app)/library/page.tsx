'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FileStack, FolderPlus, Plus, Trash2 } from 'lucide-react';
import { useAuth } from '@clerk/nextjs';
import { useRouter, useSearchParams } from 'next/navigation';
import { useQueryClient } from '@tanstack/react-query';

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
import { createFolder, createProject, deleteFile } from '@/lib/api';
import {
  useFiles,
  useFolders,
  useInvalidateFiles,
  useOnboarding,
  useProjects,
  projectKeys,
} from '@/lib/queries';
import type { FileMeta, Folder, Project } from '@/lib/types';
import { cn } from '@/lib/utils';

function fileSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export default function LibraryPage() {
  const { getToken } = useAuth();
  const router = useRouter();
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  const selectedProjectId = searchParams.get('project');
  const selectedFolderId = searchParams.get('folder');
  const fileScope = useMemo(
    () => ({ projectId: selectedProjectId, folderId: selectedFolderId }),
    [selectedFolderId, selectedProjectId],
  );
  const projectsQuery = useProjects();
  const foldersQuery = useFolders(selectedProjectId);
  const filesQuery = useFiles(fileScope);
  const invalidateFiles = useInvalidateFiles(fileScope);
  const onboarding = useOnboarding();
  const targetFile = searchParams.get('file_id') ?? searchParams.get('file');

  const files = filesQuery.data ?? [];
  const listRef = useRef<HTMLUListElement | null>(null);
  const didScrollRef = useRef<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<FileMeta | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newFolderName, setNewFolderName] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);
  const [creatingFolder, setCreatingFolder] = useState(false);
  const projects = projectsQuery.data ?? [];
  const folders = foldersQuery.data ?? [];

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

  const setProject = (projectId: string | null) => {
    const next = new URLSearchParams(searchParams.toString());
    if (projectId) next.set('project', projectId);
    else next.delete('project');
    next.delete('folder');
    router.replace(`/library?${next.toString()}`);
  };

  const setFolder = (folderId: string | null) => {
    const next = new URLSearchParams(searchParams.toString());
    if (folderId) next.set('folder', folderId);
    else next.delete('folder');
    router.replace(`/library?${next.toString()}`);
  };

  const addProject = async () => {
    if (!newProjectName.trim()) return;
    setCreatingProject(true);
    try {
      const token = (await getToken()) ?? undefined;
      const project = await createProject({ name: newProjectName.trim() }, token);
      queryClient.setQueryData<Project[]>(projectKeys.all, (old) =>
        old ? [project, ...old.filter((p) => p.id !== project.id)] : [project],
      );
      setNewProjectName('');
      setProject(project.id);
    } finally {
      setCreatingProject(false);
    }
  };

  const addFolder = async () => {
    if (!selectedProjectId || !newFolderName.trim()) return;
    setCreatingFolder(true);
    try {
      const token = (await getToken()) ?? undefined;
      const folder = await createFolder(selectedProjectId, { name: newFolderName.trim() }, token);
      queryClient.setQueryData<Folder[]>(projectKeys.folders(selectedProjectId), (old) =>
        old ? [...old.filter((f) => f.id !== folder.id), folder] : [folder],
      );
      setNewFolderName('');
    } finally {
      setCreatingFolder(false);
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
        <div className="mb-4 grid gap-3 rounded-lg border border-parchment-warm bg-parchment px-4 py-3 md:grid-cols-[1fr_1fr]">
          <div>
            <label className="text-xs font-medium text-ink-muted">Subject</label>
            <select
              value={selectedProjectId ?? ''}
              onChange={(e) => setProject(e.target.value || null)}
              className="mt-1 w-full rounded border border-parchment-warm bg-white px-2 py-2 text-sm text-ink"
            >
              <option value="">All subjects</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <div className="mt-2 flex gap-2">
              <input
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="New subject"
                className="min-w-0 flex-1 rounded border border-parchment-warm bg-white px-2 py-1.5 text-sm"
              />
              <Button size="sm" onClick={addProject} disabled={creatingProject || !newProjectName.trim()}>
                <Plus className="mr-1 h-3 w-3" />
                Add
              </Button>
            </div>
          </div>
          <div>
            <label className="text-xs font-medium text-ink-muted">Folder</label>
            <select
              value={selectedFolderId ?? ''}
              onChange={(e) => setFolder(e.target.value || null)}
              disabled={!selectedProjectId}
              className="mt-1 w-full rounded border border-parchment-warm bg-white px-2 py-2 text-sm text-ink disabled:opacity-60"
            >
              <option value="">All folders</option>
              {folders.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name}
                </option>
              ))}
            </select>
            <div className="mt-2 flex gap-2">
              <input
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="New folder"
                disabled={!selectedProjectId}
                className="min-w-0 flex-1 rounded border border-parchment-warm bg-white px-2 py-1.5 text-sm disabled:opacity-60"
              />
              <Button
                size="sm"
                onClick={addFolder}
                disabled={creatingFolder || !selectedProjectId || !newFolderName.trim()}
              >
                <FolderPlus className="mr-1 h-3 w-3" />
                Add
              </Button>
            </div>
          </div>
        </div>
        <Dropzone
          projectId={selectedProjectId}
          folderId={selectedFolderId}
          onComplete={() => void invalidateFiles()}
        />
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
