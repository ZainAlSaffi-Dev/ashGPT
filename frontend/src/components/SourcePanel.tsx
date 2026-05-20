'use client';

import { FileText, Image as ImageIcon } from 'lucide-react';

import type { SourceHit } from '@/lib/types';

interface Props {
  sources: SourceHit[];
}

export function SourcePanel({ sources }: Props) {
  if (!sources?.length) return null;
  return (
    <details className="mt-3 rounded border border-parchment-warm bg-parchment-warm/40 text-xs text-ink-muted">
      <summary className="cursor-pointer px-3 py-2 font-medium text-ink">
        {sources.length} source{sources.length === 1 ? '' : 's'}
      </summary>
      <ul className="divide-y divide-parchment-warm">
        {sources.map((s, i) => {
          const Icon = s.doc_type === 'lecture_slide' || s.doc_type === 'slide' ? ImageIcon : FileText;
          return (
            <li key={i} className="px-3 py-2">
              <div className="flex items-center gap-2 text-ink">
                <Icon className="h-3.5 w-3.5 text-accent" />
                <span className="font-medium">{s.source ?? 'unknown'}</span>
                {s.week && <span className="text-ink-soft">· {s.week}</span>}
              </div>
              <p className="mt-1 line-clamp-2 text-ink-muted">{s.snippet}</p>
            </li>
          );
        })}
      </ul>
    </details>
  );
}
