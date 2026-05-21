'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@clerk/nextjs';
import { useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  BookOpen,
  FileStack,
  FolderOpen,
  FolderPlus,
  MessageSquare,
  UploadCloud,
} from 'lucide-react';

import { Dropzone } from '@/components/Dropzone';
import { LibraryFileList } from '@/components/library/LibraryFileList';
import { ProjectSessionsPanel } from '@/components/library/ProjectSessionsPanel';
import { Button } from '@/components/ui/Button';
import { SkeletonList } from '@/components/ui/Skeleton';
import { createFolder } from '@/lib/api';
import { projectKeys, useFiles, useFolders, useInvalidateFiles, useProjects } from '@/lib/queries';
import type { FileListScope, Folder } from '@/lib/types';
import { cn } from '@/lib/utils';

const subjectPalette = ['#7a3b2e', '#315f72', '#596b3a', '#6f4d7a', '#8a6536'];

function fallbackColor(id: string) {
  const total = Array.from(id).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return subjectPalette[total % subjectPalette.length];
}

function formatUpdated(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(new Date(value));
}

export default function ProjectWorkspacePage() {
  const params = useParams<{ projectId: string }>();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { getToken } = useAuth();
  const queryClient = useQueryClient();
  const projectId = params.projectId;
  const rawFolderId = searchParams.get('folder');
  const targetFile = searchParams.get('file_id') ?? searchParams.get('file');

  const projectsQuery = useProjects();
  const foldersQuery = useFolders(projectId);
  const project = projectsQuery.data?.find((item) => item.id === projectId);
  const folders = foldersQuery.data ?? [];
  const rawFolderBelongsToProject = rawFolderId
    ? folders.some((folder) => folder.id === rawFolderId)
    : true;
  const folderSelectionPending = !!rawFolderId && foldersQuery.isLoading;
  const folderSelectionInvalid = !!rawFolderId && !foldersQuery.isLoading && !rawFolderBelongsToProject;
  const selectedFolderId =
    rawFolderId && (folderSelectionPending || rawFolderBelongsToProject)
      ? rawFolderId
      : null;
  const fileScope = useMemo<FileListScope>(
    () => ({ projectId, folderId: selectedFolderId }),
    [projectId, selectedFolderId],
  );
  const filesQuery = useFiles(fileScope);
  const invalidateFiles = useInvalidateFiles(fileScope);
  const files = filesQuery.data ?? [];
  const [newFolderName, setNewFolderName] = useState('');
  const [creatingFolder, setCreatingFolder] = useState(false);

  const selectFolder = (folderId: string | null) => {
    const next = new URLSearchParams(searchParams.toString());
    if (folderId) next.set('folder', folderId);
    else next.delete('folder');
    const query = next.toString();
    router.replace(`/library/${projectId}${query ? `?${query}` : ''}` as Route);
  };

  useEffect(() => {
    if (folderSelectionInvalid) {
      selectFolder(null);
    }
    // selectFolder closes over searchParams, so spell the stable parts out.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folderSelectionInvalid, projectId, rawFolderId]);

  const addFolder = async () => {
    if (!newFolderName.trim()) return;
    setCreatingFolder(true);
    try {
      const token = (await getToken()) ?? undefined;
      const folder = await createFolder(projectId, { name: newFolderName.trim() }, token);
      queryClient.setQueryData<Folder[]>(projectKeys.folders(projectId), (old) =>
        old ? [...old.filter((item) => item.id !== folder.id), folder] : [folder],
      );
      setNewFolderName('');
      selectFolder(folder.id);
    } finally {
      setCreatingFolder(false);
    }
  };

  if (projectsQuery.isLoading) {
    return (
      <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
        <SkeletonList rows={6} />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col items-center justify-center px-6 py-16 text-center">
        <BookOpen className="h-10 w-10 text-accent" />
        <h1 className="mt-4 font-serif text-2xl text-ink">Subject not found</h1>
        <p className="mt-2 text-sm text-ink-muted">
          This subject may have been archived or is not available to your account.
        </p>
        <Button asChild className="mt-6" variant="secondary">
          <Link href="/library">Back to Library</Link>
        </Button>
      </div>
    );
  }

  const color = project.color || fallbackColor(project.id);
  const selectedFolder = folders.find((folder) => folder.id === selectedFolderId);
  const chatHref = `/chat?project=${project.id}${
    selectedFolderId ? `&folder=${selectedFolderId}` : ''
  }` as Route;

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
      <motion.header
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22 }}
        className="border-b border-parchment-warm pb-6"
      >
        <Link
          href="/library"
          className="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-ink-muted transition hover:bg-parchment-warm hover:text-ink"
        >
          <ArrowLeft className="h-4 w-4" />
          Library
        </Link>
        <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="flex min-w-0 items-start gap-4">
            <span
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg text-white shadow-sm"
              style={{ backgroundColor: color }}
            >
              <BookOpen className="h-6 w-6" />
            </span>
            <div className="min-w-0">
              <h1 className="truncate font-serif text-3xl text-ink">{project.name}</h1>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-ink-muted">
                {project.description ||
                  'A focused workspace for this subject: upload readings, organise folders, and ask scoped questions.'}
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-ink-soft">
                <span>{files.length} visible file{files.length === 1 ? '' : 's'}</span>
                <span>{folders.length} folder{folders.length === 1 ? '' : 's'}</span>
                <span>Updated {formatUpdated(project.updated_at)}</span>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button asChild variant="secondary">
              <a href="#upload">
                <UploadCloud className="h-4 w-4" />
                Upload
              </a>
            </Button>
            <Button asChild>
              <Link href={chatHref}>
                <MessageSquare className="h-4 w-4" />
                Ask
              </Link>
            </Button>
          </div>
        </div>
      </motion.header>

      <div className="mt-6 grid gap-6 lg:grid-cols-[17rem_minmax(0,1fr)]">
        <aside className="lg:sticky lg:top-4 lg:self-start">
          <div className="rounded-lg border border-parchment-warm bg-parchment p-3">
            <div className="flex items-center justify-between gap-2 px-1">
              <div className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4 text-accent" />
                <h2 className="text-sm font-semibold text-ink">Folders</h2>
              </div>
              <span className="text-xs text-ink-soft">{folders.length}</span>
            </div>
            <div className="mt-3 flex flex-col gap-1">
              <FolderButton
                label="All files"
                active={!selectedFolderId}
                onClick={() => selectFolder(null)}
              />
              {foldersQuery.isLoading ? (
                <SkeletonList rows={3} className="mt-2" />
              ) : (
                folders.map((folder) => (
                  <FolderButton
                    key={folder.id}
                    label={folder.name}
                    active={selectedFolderId === folder.id}
                    onClick={() => selectFolder(folder.id)}
                  />
                ))
              )}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                void addFolder();
              }}
              className="mt-3 flex gap-2 border-t border-parchment-warm pt-3"
            >
              <input
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="New folder"
                className="min-w-0 flex-1 rounded-md border border-parchment-warm bg-white px-2 py-1.5 text-sm text-ink placeholder:text-ink-soft focus:border-accent focus:outline-none"
              />
              <Button
                size="icon"
                aria-label="Create folder"
                disabled={creatingFolder || !newFolderName.trim()}
              >
                <FolderPlus className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </aside>

        <main className="min-w-0">
          <ProjectSessionsPanel projectId={project.id} folderId={selectedFolderId} />

          <section>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2">
                <FileStack className="h-5 w-5 text-accent" />
                <div>
                  <h2 className="font-serif text-xl text-ink">
                    {selectedFolder ? selectedFolder.name : 'Files'}
                  </h2>
                  <p className="text-sm text-ink-muted">
                    {selectedFolder
                      ? 'Only this folder will be used for folder-scoped actions.'
                      : 'All files in this subject are visible here.'}
                  </p>
                </div>
              </div>
              <Button asChild variant="ghost" size="sm">
                <Link href={chatHref}>
                  <MessageSquare className="h-4 w-4" />
                  Ask with this scope
                </Link>
              </Button>
            </div>
            <LibraryFileList
              scope={fileScope}
              targetFile={targetFile}
              emptyTitle={selectedFolder ? 'No files in this folder' : 'No files in this subject yet'}
              emptyDescription="Upload readings below and they will stay attached to this subject scope."
            />
          </section>

          <section id="upload" className="mt-8">
            <div className="mb-3 flex items-center gap-2">
              <UploadCloud className="h-5 w-5 text-accent" />
              <h2 className="font-serif text-xl text-ink">Upload to {project.name}</h2>
            </div>
            <Dropzone
              projectId={project.id}
              folderId={selectedFolderId}
              disabled={folderSelectionPending || folderSelectionInvalid}
              disabledReason={
                folderSelectionPending
                  ? 'Checking this folder before upload'
                  : 'That folder was not found in this subject'
              }
              onComplete={() => void invalidateFiles()}
            />
          </section>
        </main>
      </div>
    </div>
  );
}

function FolderButton({
  label,
  active,
  count,
  onClick,
}: {
  label: string;
  active: boolean;
  count?: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-2 text-left text-sm transition',
        active ? 'bg-parchment-warm text-ink' : 'text-ink-muted hover:bg-parchment-warm hover:text-ink',
      )}
    >
      <span className="flex min-w-0 items-center gap-2">
        <FolderOpen className="h-4 w-4 shrink-0" />
        <span className="truncate">{label}</span>
      </span>
      {typeof count === 'number' && <span className="text-xs text-ink-soft">{count}</span>}
    </button>
  );
}
