'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';

const WelcomeTour = dynamic(
  () => import('./WelcomeTour').then((mod) => mod.WelcomeTour),
  { ssr: false },
);

export function WelcomeTourMount() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const schedule =
      window.requestIdleCallback ??
      ((cb: IdleRequestCallback) => window.setTimeout(() => cb({} as IdleDeadline), 250));
    const cancel =
      window.cancelIdleCallback ??
      ((id: number) => window.clearTimeout(id));
    const id = schedule(() => setReady(true), { timeout: 1_200 });
    return () => cancel(id);
  }, []);

  return ready ? <WelcomeTour /> : null;
}
