'use client';

import { useEffect, useRef, useState } from 'react';

interface Props {
  diagram: string;
}

/**
 * Lazy-loads mermaid and renders ``diagram`` into a div. Mermaid runs only
 * on the client (it touches the DOM), so we import dynamically.
 */
export function MermaidRenderer({ diagram }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'neutral' });
        const id = 'm' + Math.random().toString(36).slice(2);
        const { svg } = await mermaid.render(id, diagram);
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [diagram]);

  if (error) {
    return (
      <pre className="rounded bg-parchment-warm p-3 text-xs text-ink-muted">
        Mermaid render failed: {error}
        {'\n\n'}
        {diagram}
      </pre>
    );
  }

  return <div ref={ref} className="my-4 overflow-x-auto rounded bg-parchment-warm p-3" />;
}
