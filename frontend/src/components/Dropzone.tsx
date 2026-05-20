'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudUpload, Loader2 } from 'lucide-react';

import { presignUpload, processUpload, uploadBlob } from '@/lib/api';
import { cn } from '@/lib/utils';

interface Props {
  onComplete?: () => void;
}

interface UploadStatus {
  name: string;
  state: 'uploading' | 'processing' | 'ready' | 'failed';
  detail?: string;
}

const ACCEPT = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
};

export function Dropzone({ onComplete }: Props) {
  const [uploads, setUploads] = useState<UploadStatus[]>([]);

  const onDrop = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        setUploads((u) => [
          ...u,
          { name: file.name, state: 'uploading' },
        ]);
        try {
          const presign = await presignUpload({
            name: file.name,
            mime: file.type || 'application/octet-stream',
          });
          await uploadBlob(presign, file);
          setUploads((u) =>
            u.map((x) => (x.name === file.name ? { ...x, state: 'processing' } : x)),
          );
          const res = await processUpload(presign.file_id);
          setUploads((u) =>
            u.map((x) =>
              x.name === file.name
                ? {
                    ...x,
                    state: res.status === 'ready' ? 'ready' : 'failed',
                    detail:
                      res.status === 'ready'
                        ? `${res.chunk_count} chunks`
                        : res.status,
                  }
                : x,
            ),
          );
        } catch (e) {
          setUploads((u) =>
            u.map((x) =>
              x.name === file.name
                ? { ...x, state: 'failed', detail: e instanceof Error ? e.message : String(e) }
                : x,
            ),
          );
        }
      }
      onComplete?.();
    },
    [onComplete],
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
        <p className="text-xs text-ink-soft">PDF · DOCX · TXT · MD · PNG · JPG</p>
      </div>
      {uploads.length > 0 && (
        <ul className="space-y-1 text-sm">
          {uploads.map((u) => (
            <li
              key={u.name}
              className="flex items-center justify-between rounded border border-parchment-warm bg-parchment px-3 py-2"
            >
              <span className="text-ink">{u.name}</span>
              <span className="flex items-center gap-2 text-xs text-ink-muted">
                {(u.state === 'uploading' || u.state === 'processing') && (
                  <Loader2 className="h-3 w-3 animate-spin" />
                )}
                {u.state === 'uploading' && 'Uploading'}
                {u.state === 'processing' && 'Processing'}
                {u.state === 'ready' && (
                  <span className="text-accent">✓ {u.detail}</span>
                )}
                {u.state === 'failed' && (
                  <span className="text-red-600">✗ {u.detail}</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
