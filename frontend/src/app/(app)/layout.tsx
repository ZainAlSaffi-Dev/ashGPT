import { AppShell } from '@/components/AppShell';
import { Providers } from '@/components/Providers';

// Authenticated routes must not be statically prerendered — Clerk needs a
// live request context to resolve the user, and we don't ship a Clerk key
// at build time. ``force-dynamic`` opts the whole ``(app)`` group out of
// SSG/ISR and into per-request rendering.
export const dynamic = 'force-dynamic';
export const runtime = 'edge';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <Providers>
      <AppShell>{children}</AppShell>
    </Providers>
  );
}
