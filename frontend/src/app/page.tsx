import { SignInButton, SignedIn, SignedOut } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import { auth } from '@clerk/nextjs/server';
import { BookMarked } from 'lucide-react';

export default async function LandingPage() {
  const { userId } = await auth();
  if (userId) redirect('/chat');

  return (
    <main className="grid min-h-screen place-items-center bg-parchment px-6 py-16">
      <div className="max-w-xl text-center">
        <BookMarked className="mx-auto h-10 w-10 text-accent" />
        <h1 className="mt-6 font-serif text-4xl text-ink">LawGPT</h1>
        <p className="mt-4 text-ink-muted">
          Your notes, with IRAC analysis, chronology diagrams, and exam practice.
          Built for law students who want quick answers backed by their own readings.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <SignedOut>
            <SignInButton mode="modal">
              <button className="rounded-md bg-accent px-5 py-2 text-parchment shadow hover:bg-accent-hover">
                Sign in
              </button>
            </SignInButton>
          </SignedOut>
          <SignedIn>
            <a
              href="/chat"
              className="rounded-md bg-accent px-5 py-2 text-parchment shadow hover:bg-accent-hover"
            >
              Open the app
            </a>
          </SignedIn>
        </div>
      </div>
    </main>
  );
}
