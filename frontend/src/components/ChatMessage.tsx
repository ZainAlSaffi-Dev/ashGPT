'use client';

import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

import { MermaidRenderer } from './MermaidRenderer';
import { SourcePanel } from './SourcePanel';
import type { ChatTurn } from '@/lib/useChat';
import { cn, extractMermaid, withoutMermaid } from '@/lib/utils';

interface Props {
  turn: ChatTurn;
}

export function ChatMessage({ turn }: Props) {
  const isUser = turn.role === 'user';
  // Prefer the discrete `mermaid` field; fall back to inline-fence parsing.
  const mermaid = turn.mermaid || extractMermaid(turn.content);
  const bodyMd = turn.mermaid ? turn.content : withoutMermaid(turn.content);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      className={cn(
        'rounded-2xl px-4 py-3 max-w-[90%]',
        isUser
          ? 'ml-auto bg-accent text-parchment shadow-sm'
          : 'mr-auto bg-parchment text-ink ring-1 ring-parchment-warm',
      )}
    >
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
        >
          {bodyMd || ' '}
        </ReactMarkdown>
        {turn.streaming && (
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block w-[0.5ch] animate-[blink_1s_steps(1)_infinite] text-ink-muted"
          >
            ▍
          </span>
        )}
      </div>
      {mermaid && <MermaidRenderer diagram={mermaid} />}
      {turn.irac && (
        <details className="mt-3 rounded border border-parchment-warm bg-parchment-warm/40 text-xs">
          <summary className="cursor-pointer px-3 py-2 font-medium text-ink">IRAC analysis</summary>
          <pre className="whitespace-pre-wrap px-3 py-2 text-ink-muted">{turn.irac}</pre>
        </details>
      )}
      {turn.sources && <SourcePanel sources={turn.sources} />}
      {turn.verification && turn.verification.all_supported === false && (
        <p className="mt-2 text-xs text-accent">
          ⚠ Some citations could not be verified in retrieved sources.
        </p>
      )}
      {turn.latency_ms != null && !turn.streaming && (
        <p className="mt-2 text-[10px] uppercase tracking-wide text-ink-soft">
          {turn.intent ?? 'general'} · {turn.latency_ms} ms
        </p>
      )}
    </motion.div>
  );
}
