import { UserButton } from '@clerk/nextjs';

import { Sidebar } from '@/components/Sidebar';

// Authenticated routes must not be statically prerendered — Clerk needs a
// live request context to resolve the user, and we don't ship a Clerk key
// at build time. ``force-dynamic`` opts the whole ``(app)`` group out of
// SSG/ISR and into per-request rendering.
export const dynamic = 'force-dynamic';
export const runtime = 'edge';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex flex-1 flex-col">
        <header className="flex h-12 items-center justify-end border-b border-parchment-warm bg-parchment px-4">
          <UserButton />
        </header>
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
