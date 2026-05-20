import { ClerkProvider } from '@clerk/nextjs';
import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'LawGPT',
  description: 'A law-student study assistant — IRAC, chronology, summaries, exams from your own notes.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className="min-h-screen">{children}</body>
      </html>
    </ClerkProvider>
  );
}
