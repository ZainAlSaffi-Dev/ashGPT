'use client';

import { useCallback, useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import { useAuth, UserButton } from '@clerk/nextjs';
import { AnimatePresence, motion } from 'framer-motion';
import { Menu, PanelLeftClose, PanelLeftOpen } from 'lucide-react';

import { RouteTransition } from './ui/RouteTransition';
import { Sidebar } from './Sidebar';

const SIDEBAR_HIDDEN_KEY = 'ashgpt:sidebar-hidden';

export function AppShell({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [desktopHidden, setDesktopHidden] = useState(false);
  const pathname = usePathname();
  const { getToken } = useAuth();

  // Hydrate persisted desktop sidebar state.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SIDEBAR_HIDDEN_KEY);
      if (stored === '1') setDesktopHidden(true);
    } catch {
      // localStorage may be unavailable (Safari private mode); ignore.
    }
  }, []);

  // Warm Clerk's in-memory token as soon as the shell mounts so the first
  // round of useQuery calls never races sign-in hydration. Cheap belt-and-
  // braces alongside ``useAuthReady``.
  useEffect(() => {
    void getToken().catch(() => null);
  }, [getToken]);

  // Close the drawer whenever the route changes so navigating between
  // pages on mobile doesn't leave the panel sitting open over the new
  // route.
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Lock body scroll while the drawer is open — otherwise the page behind
  // the backdrop scrolls when the user drags inside the sidebar.
  useEffect(() => {
    if (!mobileOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [mobileOpen]);

  // Close on Escape so keyboard users aren't trapped in the drawer.
  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mobileOpen]);

  const close = useCallback(() => setMobileOpen(false), []);

  const toggleDesktopSidebar = useCallback(() => {
    setDesktopHidden((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(SIDEBAR_HIDDEN_KEY, next ? '1' : '0');
      } catch {
        // ignore
      }
      return next;
    });
  }, []);

  return (
    <div className="flex h-screen">
      {/* Desktop sidebar — animated collapse on md+. */}
      <motion.div
        className="hidden overflow-hidden md:flex"
        initial={false}
        animate={{ width: desktopHidden ? 0 : 240 }}
        transition={{ type: 'spring', stiffness: 320, damping: 34 }}
      >
        <Sidebar />
      </motion.div>

      {/* Mobile drawer + backdrop. */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              onClick={close}
              className="fixed inset-0 z-40 bg-ink/40 md:hidden"
              aria-hidden="true"
            />
            <motion.div
              key="drawer"
              role="dialog"
              aria-modal="true"
              aria-label="Navigation"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'tween', ease: 'easeOut', duration: 0.22 }}
              className="fixed inset-y-0 left-0 z-50 flex md:hidden"
            >
              <Sidebar />
            </motion.div>
          </>
        )}
      </AnimatePresence>

      <div className="flex flex-1 flex-col">
        <header className="flex h-12 items-center justify-between gap-2 border-b border-parchment-warm bg-parchment px-3 md:px-4">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="-ml-1 rounded p-1.5 text-ink-muted transition hover:bg-parchment-warm hover:text-ink md:hidden"
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={toggleDesktopSidebar}
            className="hidden rounded p-1.5 text-ink-muted transition hover:bg-parchment-warm hover:text-ink md:inline-flex"
            aria-label={desktopHidden ? 'Show sidebar' : 'Hide sidebar'}
            title={desktopHidden ? 'Show sidebar' : 'Hide sidebar'}
          >
            {desktopHidden ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <PanelLeftClose className="h-4 w-4" />
            )}
          </button>
          <div className="ml-auto">
            <UserButton />
          </div>
        </header>
        <main className="flex-1 overflow-y-auto">
          <RouteTransition>{children}</RouteTransition>
        </main>
      </div>
    </div>
  );
}
