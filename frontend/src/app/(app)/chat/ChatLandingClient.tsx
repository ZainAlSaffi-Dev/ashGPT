'use client';

import { useSearchParams } from 'next/navigation';

import { ChatSurface } from '@/components/ChatSurface';
import { scopeFromSearchParams } from '@/lib/scope';

export function ChatLandingClient() {
  const params = useSearchParams();
  return <ChatSurface scope={scopeFromSearchParams(params)} />;
}
