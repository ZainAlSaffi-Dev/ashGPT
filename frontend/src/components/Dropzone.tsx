'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudUpload, Loader2, RotateCcw } from 'lucide-react';
import { useAuth } from '@clerk/nextjs';

import { presignUpload, processUpload, uploadBlob } from '@/lib/api';
import { cn } from '@/lib/utils';

interface Props {
  onComplete?: () => void;
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

// Cap aligned with the typical Cloudflare Worker request budget — the
// browser PUT goes direct to R2 so this is just a sanity guard against
// uploading a 5 GB book by accident.
const MAX_BYTES = 100 * 1024 * 1024;

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
  if (msg.startsWith('API ')) return msg;
  return msg;
}

export function Dropzone({ onComplete }: Props) {
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
        const token = (await getToken()) ?? undefined;
        const presign = await presignUpload(
          {
            name: file.name,
            mime: file.type || 'application/octet-stream',
          },
          token,
        );
        await uploadBlob(presign, file, token);
        // Brief verifying state so the user sees the upload landed before
        // ingestion latency takes over.
        update(id, { state: 'verifying', progress: 100 });
        const res = await processUpload(presign.file_id, token);
        if (res.status === 'ready') {
          update(id, { state: 'ready', detail: `${res.chunk_count} chunks indexed` });
        } else if (res.status === 'queued') {
          update(id, { state: 'processing', detail: 'queued for ingestion' });
        } else {
          // status === 'failed' or unknown
          update(id, { state: 'failed', detail: res.status });
        }
      } catch (e) {
        update(id, { state: 'failed', detail: humanError(e) });
      }
    },
    [getToken, update],
  );

  const onDrop = useCallback(
    async (files: File[]) => {
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
    [onComplete, runUpload],
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
    accept: ACCEPT,
    multiple: true,
  });

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={cn(
          'flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-parchment-warm bg-parchment-warm/30 px-6 py-12 text-center transition',
          isDragActive && 'border-accent bg-parchment-warm/60',
        )}
      >
        <input {...getInputProps()} />
        <CloudUpload className="h-8 w-8 text-accent" />
        <p className="font-serif text-ink">
          {isDragActive ? 'Drop the files here' : 'Drop PDFs, DOCX, images, or notes here'}
        </p>
        <p className="text-xs text-ink-soft">PDF · DOCX · TXT · MD · PNG · JPG (max 100 MB)</p>
      </div>
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
