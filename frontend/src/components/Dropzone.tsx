'use client';

import { useCallback, useMemo, useState } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { CloudUpload, Loader2, RotateCcw } from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuth } from '@clerk/nextjs';

import { deleteFile, presignUpload, processUpload, uploadBlob, withAuth } from '@/lib/api';
import { cn } from '@/lib/utils';

interface Props {
  onComplete?: () => void;
  projectId?: string | null;
  folderId?: string | null;
  disabled?: boolean;
  disabledReason?: string;
}

type UploadState =
  | 'queued'
  | 'uploading'
  | 'verifying'
  | 'processing'
  | 'ready'
  | 'failed';

interface UploadStatus {
  id: string;
  file: File;
  state: UploadState;
  detail?: string;
  progress?: number; // 0-100 during 'uploading'
}

const ACCEPT = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
};

// Cap aligned with the Worker upload endpoint and the UX we can process
// predictably in one study-session upload.
const MAX_BYTES = 100 * 1024 * 1024;

const MIME_BY_EXTENSION: Record<string, string> = {
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  txt: 'text/plain',
  md: 'text/markdown',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
};

function uploadMime(file: File): string {
  if (file.type && file.type !== 'application/octet-stream') return file.type;
  const extension = file.name.split('.').pop()?.toLowerCase();
  return (extension && MIME_BY_EXTENSION[extension]) || file.type || 'application/octet-stream';
}

function humanError(err: unknown): string {
  if (!(err instanceof Error)) return String(err);
  const msg = err.message;
  // Surface the most common failure modes in plain language so the user
  // knows whether to retry or fix something.
  if (msg.includes('Failed to fetch') || msg.toLowerCase().includes('networkerror')) {
    return 'Network blocked the upload (likely CORS or offline). Try again.';
  }
  if (msg.includes('API 409')) return 'Server didn’t see the file in R2. Try again.';
  if (msg.includes('API 413')) return 'File is too large for the backend.';
  if (msg.includes('API 401')) return 'Sign-in expired. Refresh and retry.';
  if (err.name === 'AuthNotReadyError' || msg.includes('Auth not ready')) {
    return 'Still connecting to sign-in. Try again in a moment.';
  }
  if (err.name === 'AbortError') return 'Upload timed out. Try again.';
  if (msg.includes('folder not found')) return 'That folder no longer exists. Pick another folder and retry.';
  if (msg.includes('folder does not belong to project')) {
    return 'That folder is not in this subject. Pick another folder and retry.';
  }
  if (msg.startsWith('API ')) return msg;
  return msg;
}

function rejectionMessage(file: File, errors: readonly { code: string; message: string }[]): string {
  const tooLarge = errors.some((error) => error.code === 'file-too-large');
  if (tooLarge) return `${file.name} is larger than the 100 MB limit.`;
  const invalidType = errors.some((error) => error.code === 'file-invalid-type');
  if (invalidType) return `${file.name} is not a supported file type.`;
  return `${file.name}: ${errors.map((error) => error.message).join(', ')}`;
}

export function Dropzone({
  onComplete,
  projectId = null,
  folderId = null,
  disabled = false,
  disabledReason,
}: Props) {
  const { getToken } = useAuth();
  const [uploads, setUploads] = useState<UploadStatus[]>([]);

  const update = useCallback((id: string, patch: Partial<UploadStatus>) => {
    setUploads((u) => u.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  }, []);

  const runUpload = useCallback(
    async (item: UploadStatus) => {
      const { id, file } = item;
      update(id, { state: 'uploading', progress: 0, detail: undefined });
      try {
        if (file.size > MAX_BYTES) {
          throw new Error(`File too large (${Math.round(file.size / 1024 / 1024)} MB > 100 MB)`);
        }
        await withAuth(getToken, async (token) => {
          const mime = uploadMime(file);
          const presign = await presignUpload(
            {
              name: file.name,
              mime,
              project_id: projectId,
              folder_id: folderId,
            },
            token,
          );
          try {
            await uploadBlob(presign, file, token, mime);
          } catch (uploadError) {
            await deleteFile(presign.file_id, token).catch(() => undefined);
            throw uploadError;
          }
          // Brief verifying state so the user sees the upload landed before
          // ingestion latency takes over.
          update(id, { state: 'verifying', progress: 100 });
          const res = await processUpload(presign.file_id, token);
          if (res.status === 'ready') {
            update(id, { state: 'ready', detail: `${res.chunk_count} chunks indexed` });
          } else if (res.status === 'queued' || res.status === 'processing') {
            update(id, {
              state: 'processing',
              detail: res.status === 'queued' ? 'queued for ingestion' : 'indexing',
            });
          } else {
            // status === 'failed' or unknown
            update(id, { state: 'failed', detail: res.status });
          }
        });
      } catch (e) {
        update(id, { state: 'failed', detail: humanError(e) });
      }
    },
    [folderId, getToken, projectId, update],
  );

  const onDrop = useCallback(
    async (files: File[]) => {
      if (disabled) return;
      const items: UploadStatus[] = files.map((file) => ({
        id: crypto.randomUUID(),
        file,
        state: 'queued',
      }));
      setUploads((u) => [...u, ...items]);
      // Parallelism kept low to avoid burning Worker concurrency.
      const POOL = 2;
      const queue = [...items];
      const workers = Array.from({ length: Math.min(POOL, queue.length) }, async () => {
        while (queue.length) {
          const next = queue.shift();
          if (!next) break;
          await runUpload(next);
        }
      });
      await Promise.all(workers);
      onComplete?.();
    },
    [disabled, onComplete, runUpload],
  );

  const onDropRejected = useCallback(
    (rejections: FileRejection[]) => {
      const items: UploadStatus[] = rejections.map(({ file, errors }) => ({
        id: crypto.randomUUID(),
        file,
        state: 'failed',
        detail: rejectionMessage(file, errors),
      }));
      setUploads((u) => [...u, ...items]);
    },
    [],
  );

  const retry = useCallback(
    async (id: string) => {
      const item = uploads.find((u) => u.id === id);
      if (!item) return;
      await runUpload(item);
      onComplete?.();
    },
    [uploads, runUpload, onComplete],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: ACCEPT,
    disabled,
    maxSize: MAX_BYTES,
    multiple: true,
  });

  // Aggregate progress across the batch. ``done`` covers terminal states
  // (ready + failed) so a failure still moves the bar forward — the user
  // can retry the row directly.
  const summary = useMemo(() => {
    if (uploads.length === 0) return null;
    const total = uploads.length;
    const ready = uploads.filter((u) => u.state === 'ready').length;
    const failed = uploads.filter((u) => u.state === 'failed').length;
    const processing = uploads.filter((u) => u.state === 'processing').length;
    const inflight = uploads.filter(
      (u) => u.state === 'queued' || u.state === 'uploading' || u.state === 'verifying',
    ).length;
    const done = total - inflight;
    const pct = Math.round((done / total) * 100);
    return { total, ready, failed, processing, inflight, pct };
  }, [uploads]);

  const clearFinished = useCallback(() => {
    setUploads((u) =>
      u.filter((x) => x.state === 'queued' || x.state === 'uploading' || x.state === 'verifying'),
    );
  }, []);

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={cn(
          'flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-parchment-warm bg-parchment-warm/30 px-6 py-12 text-center transition',
          isDragActive && 'border-accent bg-parchment-warm/60',
          disabled && 'cursor-not-allowed opacity-60',
        )}
      >
        <input {...getInputProps()} />
        <CloudUpload className="h-8 w-8 text-accent" />
        <p className="font-serif text-ink">
          {disabled
            ? disabledReason || 'Uploads are temporarily unavailable'
            : isDragActive
              ? 'Drop the files here'
              : 'Drop PDFs, DOCX, images, or notes here'}
        </p>
        <p className="text-xs text-ink-soft">PDF · DOCX · TXT · MD · PNG · JPG (max 100 MB)</p>
      </div>
      {summary && (
        <div className="rounded-lg border border-parchment-warm bg-parchment px-3 py-2">
          <div className="flex items-center justify-between text-xs text-ink-muted">
            <span className="font-medium text-ink">
              {summary.ready === summary.total
                ? `All ${summary.total} file${summary.total === 1 ? '' : 's'} ready`
                : `${summary.ready} of ${summary.total} ready`}
              {summary.failed > 0 && (
                <span className="ml-2 text-red-600">· {summary.failed} failed</span>
              )}
            </span>
            <div className="flex items-center gap-3">
              {summary.inflight > 0 && (
                <span className="flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {summary.inflight} in flight
                </span>
              )}
              {summary.inflight === 0 && (
                <button
                  type="button"
                  onClick={clearFinished}
                  className="rounded border border-parchment-warm px-1.5 py-0.5 text-[11px] text-ink-muted transition hover:bg-parchment-warm hover:text-ink"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-parchment-warm">
            <motion.div
              className={cn(
                'h-full rounded-full',
                summary.failed > 0 && summary.inflight === 0
                  ? 'bg-red-400'
                  : 'bg-accent',
              )}
              initial={{ width: 0 }}
              animate={{ width: `${summary.pct}%` }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
            />
          </div>
        </div>
      )}
      {uploads.length > 0 && (
        <ul className="space-y-1 text-sm">
          {uploads.map((u) => (
            <li
              key={u.id}
              className="flex items-center justify-between gap-3 rounded border border-parchment-warm bg-parchment px-3 py-2"
            >
              <span className="truncate text-ink" title={u.file.name}>
                {u.file.name}
              </span>
              <span className="flex shrink-0 items-center gap-2 text-xs text-ink-muted">
                {(u.state === 'queued' ||
                  u.state === 'uploading' ||
                  u.state === 'verifying' ||
                  u.state === 'processing') && <Loader2 className="h-3 w-3 animate-spin" />}
                {u.state === 'queued' && 'Queued'}
                {u.state === 'uploading' && 'Uploading'}
                {u.state === 'verifying' && 'Verifying upload'}
                {u.state === 'processing' && (u.detail ?? 'Processing')}
                {u.state === 'ready' && (
                  <span className="text-accent">✓ {u.detail}</span>
                )}
                {u.state === 'failed' && (
                  <>
                    <span className="text-red-600">✗ {u.detail}</span>
                    <button
                      type="button"
                      onClick={() => retry(u.id)}
                      className="flex items-center gap-1 rounded border border-parchment-warm px-1.5 py-0.5 text-ink-muted transition hover:bg-parchment-warm hover:text-ink"
                      title="Retry"
                    >
                      <RotateCcw className="h-3 w-3" />
                      Retry
                    </button>
                  </>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
