import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Pluck a Mermaid diagram block out of a markdown-fenced final answer. */
export function extractMermaid(text: string): string | null {
  const m = text.match(/```mermaid\n([\s\S]*?)```/);
  return m ? m[1].trim() : null;
}

/** Strip the Mermaid fence so the rest of the answer can render as markdown. */
export function withoutMermaid(text: string): string {
  return text.replace(/```mermaid\n[\s\S]*?```/g, '').trim();
}
