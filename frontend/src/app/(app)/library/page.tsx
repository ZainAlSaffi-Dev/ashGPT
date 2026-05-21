'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@clerk/nextjs';
import { useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  BookOpen,
  FileStack,
  FolderOpen,
  MessageSquare,
  Plus,
  Search,
  UploadCloud,
} from 'lucide-react';

import { Dropzone } from '@/components/Dropzone';
import { LibraryFileList } from '@/components/library/LibraryFileList';
import { OnboardingChecklist } from '@/components/OnboardingChecklist';
import { Button } from '@/components/ui/Button';
import { SkeletonList } from '@/components/ui/Skeleton';
import { createProject } from '@/lib/api';
import { projectKeys, useFiles, useInvalidateFiles, useOnboarding, useProjects } from '@/lib/queries';
import type { Project } from '@/lib/types';

const subjectPalette = ['#7a3b2e', '#315f72', '#596b3a', '#6f4d7a', '#8a6536'];

function subjectColor(project: Project, index: number) {
  return project.color || subjectPalette[index % subjectPalette.length];
}

function formatUpdated(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(new Date(value));
}

export default function LibraryPage() {
  const { getToken } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const projectsQuery = useProjects();
  const filesQuery = useFiles();
  const invalidateFiles = useInvalidateFiles();
  const onboarding = useOnboarding();
  const targetFile = searchParams.get('file_id') ?? searchParams.get('file');

  const [newProjectName, setNewProjectName] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);
  const projects = projectsQuery.data ?? [];
  const files = filesQuery.data ?? [];

  const fileCountsByProject = useMemo(() => {
    const counts = new Map<string, number>();
    for (const file of files) {
      if (!file.project_id) continue;
      counts.set(file.project_id, (counts.get(file.project_id) ?? 0) + 1);
    }
    return counts;
  }, [files]);

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
      router.push(`/library/${project.id}` as Route);
    } finally {
      setCreatingProject(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22 }}
        className="flex flex-col gap-5 border-b border-parchment-warm pb-6 lg:flex-row lg:items-end lg:justify-between"
      >
        <div>
          <div className="inline-flex items-center gap-2 rounded-md bg-parchment-warm px-2.5 py-1 text-xs font-medium text-accent">
            <BookOpen className="h-3.5 w-3.5" />
            Study library
          </div>
          <h1 className="mt-3 font-serif text-3xl text-ink">Library</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-ink-muted">
            Organise readings by subject, upload once, and start scoped chats with
            the files that belong to the workspace you are studying.
          </p>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            void addProject();
          }}
          className="flex w-full gap-2 rounded-lg border border-parchment-warm bg-parchment p-2 shadow-sm lg:max-w-md"
        >
          <input
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="Create a subject"
            className="min-w-0 flex-1 rounded-md border border-transparent bg-white px-3 py-2 text-sm text-ink placeholder:text-ink-soft focus:border-accent focus:outline-none"
          />
          <Button size="sm" disabled={creatingProject || !newProjectName.trim()}>
            <Plus className="h-3.5 w-3.5" />
            Add
          </Button>
        </form>
      </motion.div>

      {!onboarding.isComplete && (
        <div className="mt-6">
          <OnboardingChecklist variant="full" />
        </div>
      )}

      <section className="mt-7">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h2 className="font-serif text-xl text-ink">Subjects</h2>
            <p className="text-sm text-ink-muted">Click a subject to open its workspace.</p>
          </div>
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 rounded-md px-2.5 py-2 text-sm text-ink-muted transition hover:bg-parchment-warm hover:text-ink"
          >
            <MessageSquare className="h-4 w-4" />
            New chat
          </Link>
        </div>

        {projectsQuery.isLoading ? (
          <SkeletonList rows={3} className="mt-3" />
        ) : projects.length === 0 ? (
          <EmptySubjects />
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {projects.map((project, index) => (
              <SubjectCard
                key={project.id}
                project={project}
                color={subjectColor(project, index)}
                fileCount={fileCountsByProject.get(project.id) ?? 0}
              />
            ))}
          </div>
        )}
      </section>

      <section className="mt-9 grid gap-5 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div>
          <div className="flex items-center gap-2">
            <FileStack className="h-5 w-5 text-accent" />
            <h2 className="font-serif text-xl text-ink">All files</h2>
          </div>
          <LibraryFileList
            targetFile={targetFile}
            emptyDescription="Upload readings here if they do not belong to a subject yet."
          />
        </div>

        <div id="upload" className="lg:sticky lg:top-4 lg:self-start">
          <div className="mb-3 flex items-center gap-2">
            <UploadCloud className="h-5 w-5 text-accent" />
            <h2 className="font-serif text-xl text-ink">Upload</h2>
          </div>
          <Dropzone onComplete={() => void invalidateFiles()} />
        </div>
      </section>
    </div>
  );
}

function SubjectCard({
  project,
  color,
  fileCount,
}: {
  project: Project;
  color: string;
  fileCount: number;
}) {
  const href = `/library/${project.id}` as Route;
  const chatHref = `/chat?project=${project.id}` as Route;
  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="group rounded-lg border border-parchment-warm bg-parchment shadow-sm transition hover:-translate-y-0.5 hover:border-accent/30 hover:shadow-md"
    >
      <Link href={href} className="block p-4">
        <div className="flex items-start justify-between gap-3">
          <span
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md text-white shadow-sm"
            style={{ backgroundColor: color }}
          >
            <BookOpen className="h-5 w-5" />
          </span>
          <span className="rounded-md bg-white px-2 py-1 text-xs text-ink-soft ring-1 ring-parchment-warm">
            {fileCount} file{fileCount === 1 ? '' : 's'}
          </span>
        </div>
        <h3 className="mt-4 truncate font-serif text-lg text-ink">{project.name}</h3>
        <p className="mt-1 line-clamp-2 min-h-10 text-sm leading-5 text-ink-muted">
          {project.description || 'A focused workspace for readings, notes, folders, and scoped chat.'}
        </p>
        <div className="mt-4 flex items-center justify-between gap-3 text-xs text-ink-soft">
          <span>Updated {formatUpdated(project.updated_at)}</span>
          <span className="inline-flex items-center gap-1 text-accent opacity-0 transition group-hover:opacity-100">
            Open
            <FolderOpen className="h-3.5 w-3.5" />
          </span>
        </div>
      </Link>
      <div className="border-t border-parchment-warm px-4 py-2">
        <Link
          href={chatHref}
          className="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-ink-muted transition hover:bg-parchment-warm hover:text-ink"
        >
          <MessageSquare className="h-4 w-4" />
          Ask in this subject
        </Link>
      </div>
    </motion.article>
  );
}

function EmptySubjects() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      className="rounded-lg border border-dashed border-parchment-warm bg-parchment px-6 py-10 text-center"
    >
      <Search className="mx-auto h-8 w-8 text-accent" />
      <p className="mt-3 font-serif text-lg text-ink">No subjects yet</p>
      <p className="mx-auto mt-1 max-w-md text-sm text-ink-muted">
        Create a subject for each course or assignment, then upload readings into
        that workspace so chat retrieval stays focused.
      </p>
    </motion.div>
  );
}
