'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BookMarked, FileStack, GraduationCap, MessageSquare, Settings } from 'lucide-react';

import { cn } from '@/lib/utils';

const nav = [
  { href: '/chat', label: 'Chat', Icon: MessageSquare },
  { href: '/library', label: 'Library', Icon: FileStack },
  { href: '/exam', label: 'Exam', Icon: GraduationCap },
  { href: '/settings', label: 'Settings', Icon: Settings },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-parchment-warm bg-parchment p-4">
      <Link href="/" className="mb-8 flex items-center gap-2 font-serif text-xl text-ink">
        <BookMarked className="h-5 w-5 text-accent" />
        LawGPT
      </Link>
      <nav className="flex flex-col gap-1">
        {nav.map(({ href, label, Icon }) => {
          const active = pathname === href || pathname.startsWith(href + '/');
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition',
                active
                  ? 'bg-parchment-warm text-ink'
                  : 'text-ink-muted hover:bg-parchment-warm hover:text-ink',
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
