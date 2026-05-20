'use client';

import { useCallback, useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { useAuth } from '@clerk/nextjs';

import { Dropzone } from '@/components/Dropzone';
import { deleteFile, listFiles } from '@/lib/api';
import type { FileMeta } from '@/lib/types';
import { cn } from '@/lib/utils';

export default function LibraryPage() {
  const { getToken } = useAuth();
  const [files, setFiles] = useState<FileMeta[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const token = (await getToken()) ?? undefined;
      const list = await listFiles(token);
      setFiles(list);
    } finally {
      setLoading(false);
    }
  }, [getToken]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onDelete = async (id: string) => {
    if (!confirm('Delete this file and all its chunks? This cannot be undone.')) return;
    const token = (await getToken()) ?? undefined;
    await deleteFile(id, token);
    await refresh();
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <h1 className="font-serif text-2xl text-ink">Library</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Drop your readings, lecture slides, and notes. They'll be chunked,
        embedded, and made searchable for chat + exam generation.
      </p>

      <div className="mt-6">
        <Dropzone onComplete={refresh} />
      </div>

      <h2 className="mt-8 font-serif text-lg text-ink">Your files</h2>
      {loading ? (
        <p className="mt-3 text-sm text-ink-muted">Loading…</p>
      ) : files.length === 0 ? (
        <p className="mt-3 text-sm text-ink-muted">No files yet.</p>
      ) : (
        <ul className="mt-3 divide-y divide-parchment-warm rounded-lg border border-parchment-warm bg-parchment">
          {files.map((f) => (
            <li key={f.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="text-ink">{f.name}</p>
                <p className="text-xs text-ink-soft">
                  {f.doc_type}
                  {f.week ? ` · ${f.week}` : ''}
                  {' · '}
                  {f.chunk_count} chunk{f.chunk_count === 1 ? '' : 's'}
                  {' · '}
                  <span className={cn(
                    f.status === 'ready' && 'text-accent',
                    f.status === 'failed' && 'text-red-600',
                  )}>{f.status}</span>
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
          ))}
        </ul>
      )}
    </div>
  );
}
