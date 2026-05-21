'use client';

import { useEffect, useRef, useState } from 'react';

interface Props {
  diagram: string;
}

interface MermaidApi {
  initialize: (options: Record<string, unknown>) => void;
  render: (id: string, diagram: string) => Promise<{ svg: string }>;
}

declare global {
  interface Window {
    mermaid?: MermaidApi;
  }
}

const MERMAID_SRC = 'https://cdn.jsdelivr.net/npm/mermaid@11.15.0/dist/mermaid.min.js';
let mermaidLoad: Promise<MermaidApi> | null = null;

function loadMermaid(): Promise<MermaidApi> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Mermaid can only render in the browser'));
  }
  if (window.mermaid) return Promise.resolve(window.mermaid);
  if (mermaidLoad) return mermaidLoad;

  mermaidLoad = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = MERMAID_SRC;
    script.async = true;
    script.onload = () => {
      if (window.mermaid) resolve(window.mermaid);
      else {
        mermaidLoad = null;
        reject(new Error('Mermaid script loaded without exposing window.mermaid'));
      }
    };
    script.onerror = () => {
      mermaidLoad = null;
      reject(new Error('Could not load Mermaid renderer'));
    };
    document.head.appendChild(script);
  });
  return mermaidLoad;
}

/**
 * Loads Mermaid in the browser and renders ``diagram`` into a div. Keeping
 * Mermaid off the app bundle matters on Cloudflare Pages: the deprecated
 * next-on-pages adapter otherwise pulls this client-only package into the
 * Pages Function bundle and can trip publish-size failures.
 */
export function MermaidRenderer({ diagram }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    (async () => {
      try {
        const mermaid = await loadMermaid();
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
