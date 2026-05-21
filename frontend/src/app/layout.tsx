import { ClerkProvider } from '@clerk/nextjs';
import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'ashGPT',
  description: 'A law-student study assistant — IRAC, chronology, summaries, exams from your own notes.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
  const tree = (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
  // ClerkProvider throws at build time when the publishable key is missing.
  // CI / fresh CF Pages preview builds run without secrets; we skip the
  // provider in that case so the build can complete. Production sets
  // NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY via Pages env vars.
  if (!clerkKey) return tree;
  return <ClerkProvider publishableKey={clerkKey}>{tree}</ClerkProvider>;
}
