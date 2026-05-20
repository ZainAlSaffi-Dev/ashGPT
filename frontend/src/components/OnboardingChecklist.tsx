'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Check,
  FileStack,
  GraduationCap,
  MessageSquare,
} from 'lucide-react';

import { useOnboarding } from '@/lib/queries';
import { cn } from '@/lib/utils';

interface Step {
  key: number;
  label: string;
  description: string;
  href: '/library' | '/chat' | '/exam';
  Icon: React.ComponentType<{ className?: string }>;
}

const STEPS: Step[] = [
  {
    key: 1,
    label: 'Add your first file',
    description: 'Drop a PDF, DOCX, image, or set of notes — anything you study from.',
    href: '/library',
    Icon: FileStack,
  },
  {
    key: 2,
    label: 'Ask your first question',
    description: 'Try “Summarise the main argument of …” or “Explain the ratio in …”.',
    href: '/chat',
    Icon: MessageSquare,
  },
  {
    key: 3,
    label: 'Generate a practice exam',
    description: 'Mix multiple-choice and short-answer prompts grounded in your library.',
    href: '/exam',
    Icon: GraduationCap,
  },
];

interface Props {
  variant?: 'full' | 'compact';
  className?: string;
}

/** Onboarding walkthrough — three guided steps. Hides itself once complete.
 *  ``full`` is the home/landing variant with descriptions. ``compact`` is the
 *  inline-banner variant used inside other pages. */
export function OnboardingChecklist({ variant = 'full', className }: Props) {
  const state = useOnboarding();
  if (state.isLoading) return null;
  if (state.isComplete) return null;

  const currentStep = state.step;

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={cn(
        'rounded-2xl border border-parchment-warm bg-parchment shadow-sm',
        variant === 'full' ? 'p-6' : 'p-4',
        className,
      )}
    >
      <header className="mb-4 flex items-baseline justify-between">
        <h2 className="font-serif text-lg text-ink">
          {variant === 'full' ? 'Welcome — get set up in three steps' : 'Finish setup'}
        </h2>
        <span className="text-xs text-ink-soft">
          {currentStep === 'done' ? 'Done' : `Step ${currentStep} of 3`}
        </span>
      </header>

      <ol className="space-y-3">
        {STEPS.map((step) => {
          const isDone =
            (step.key === 1 && state.readyFilesCount > 0) ||
            (step.key === 2 && state.sessionsCount > 0) ||
            (step.key === 3 && state.readyFilesCount > 0 && state.sessionsCount > 0);
          const isCurrent = step.key === currentStep;
          const isLocked = step.key > (typeof currentStep === 'number' ? currentStep : 0);

          return (
            <li key={step.key}>
              <Link
                href={step.href}
                aria-disabled={isLocked}
                tabIndex={isLocked ? -1 : 0}
                className={cn(
                  'group flex items-start gap-3 rounded-xl border p-3 transition',
                  isDone && 'border-accent/40 bg-parchment-warm/40',
                  isCurrent && !isDone && 'border-accent bg-parchment-warm/70 shadow-sm',
                  isLocked &&
                    'pointer-events-none cursor-not-allowed border-parchment-warm/60 opacity-60',
                  !isDone && !isCurrent && !isLocked && 'border-parchment-warm hover:bg-parchment-warm/40',
                )}
              >
                <div
                  className={cn(
                    'mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold',
                    isDone
                      ? 'bg-accent text-parchment'
                      : isCurrent
                      ? 'bg-accent/15 text-accent ring-1 ring-accent'
                      : 'bg-parchment-warm text-ink-muted',
                  )}
                >
                  {isDone ? <Check className="h-3.5 w-3.5" /> : step.key}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <step.Icon
                      className={cn(
                        'h-4 w-4',
                        isDone ? 'text-accent' : 'text-ink-muted',
                      )}
                    />
                    <p className="text-sm font-medium text-ink">{step.label}</p>
                  </div>
                  {variant === 'full' && (
                    <p className="mt-1 text-xs text-ink-muted">{step.description}</p>
                  )}
                </div>
                {!isLocked && !isDone && (
                  <span className="self-center text-xs font-medium text-accent group-hover:underline">
                    {isCurrent ? 'Start →' : 'Open →'}
                  </span>
                )}
              </Link>
            </li>
          );
        })}
      </ol>
    </motion.section>
  );
}

/** Inline banner used when a route's primary action is gated on onboarding.
 *  Renders nothing once onboarding completes. */
export function OnboardingGate({
  requiresFiles,
  children,
}: {
  /** When true, render the gate iff the user has no ``ready`` files. */
  requiresFiles?: boolean;
  children?: React.ReactNode;
}) {
  const state = useOnboarding();
  if (state.isLoading) return null;
  if (requiresFiles && state.readyFilesCount > 0) {
    return null;
  }
  if (!requiresFiles && state.isComplete) {
    return null;
  }
  return (
    <div className="mb-6">
      <OnboardingChecklist variant="compact" />
      {children}
    </div>
  );
}
