import { AppShell } from '@/components/AppShell';
import { Providers } from '@/components/Providers';
import { WelcomeTourMount } from '@/components/onboarding/WelcomeTourMount';

// Edge runtime so Clerk's middleware/cookies are available, but no
// ``dynamic = 'force-dynamic'``: the layout itself never reads
// request-scoped state (the Clerk hooks inside ``Providers`` and
// ``AppShell`` are client-side). Marking it dynamic forced Next to
// invalidate the whole layout shell on every sibling-route nav, which
// re-evaluated ``Providers`` and briefly dropped the persisted query
// cache — visible as the history sidebar reloading every time you
// switched tabs. Letting Next reuse the layout means navigating between
// /chat, /library, /exam, /settings is a pure client-side segment swap.
export const runtime = 'edge';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <Providers>
      <AppShell>{children}</AppShell>
      <WelcomeTourMount />
    </Providers>
  );
}
