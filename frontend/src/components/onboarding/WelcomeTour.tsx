'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowRight,
  BookMarked,
  FileStack,
  GraduationCap,
  MessageSquare,
  Sparkles,
} from 'lucide-react';

import { Button } from '@/components/ui/Button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/Dialog';
import { useCurrentUser, useMarkOnboarded } from '@/lib/queries';

interface Step {
  Icon: typeof MessageSquare;
  title: string;
  body: string;
  cta?: { label: string; href?: string };
}

const STEPS: Step[] = [
  {
    Icon: Sparkles,
    title: 'Welcome to ashGPT',
    body: 'Your readings, statutes, and past papers — grounded answers with citations, IRAC analysis, chronology diagrams, and exam practice.',
  },
  {
    Icon: FileStack,
    title: 'Upload your readings',
    body: 'Drop PDFs, lecture slides, case notes, or past papers into the Library. ashGPT chunks and embeds them so every answer can cite back to the source.',
    cta: { label: 'Open the Library', href: '/library' },
  },
  {
    Icon: MessageSquare,
    title: 'Ask grounded questions',
    body: 'Ask anything — "What is adverse possession in QLD?", "Summarise Donoghue v Stevenson", "Build a chronology for module 3". Every answer is backed by your own files.',
    cta: { label: 'Start a chat', href: '/chat' },
  },
  {
    Icon: GraduationCap,
    title: 'Generate exams',
    body: 'Turn your readings into multiple-choice + short-answer practice. ashGPT grades short-answers against a rubric so you can see what you missed.',
    cta: { label: 'Try Exam mode', href: '/exam' },
  },
];

/** First-run guided tour. Mounted globally inside the (app) shell.
 *
 *  Opens once when the current user has ``onboarded_at === null``.
 *  Both "Done" and "Skip tour" POST ``/users/me/onboarded`` so the tour
 *  never reappears on the same Clerk account — even across devices.
 */
export function WelcomeTour() {
  const { data: user, isFetched } = useCurrentUser();
  const markOnboarded = useMarkOnboarded();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);

  // Open exactly once per "fresh user" page load. We only auto-open here;
  // a settings entry could re-open it manually in a future iteration.
  useEffect(() => {
    if (!isFetched || !user) return;
    if (user.onboarded_at === null) setOpen(true);
  }, [isFetched, user]);

  const finish = () => {
    setOpen(false);
    if (user?.onboarded_at === null) markOnboarded.mutate();
  };

  const next = () => {
    if (step < STEPS.length - 1) setStep((s) => s + 1);
    else finish();
  };

  const goToCta = () => {
    const cta = STEPS[step].cta;
    if (cta?.href) {
      finish();
      router.push(cta.href as never);
    } else {
      next();
    }
  };

  const Current = STEPS[step];

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) finish();
      }}
    >
      <DialogContent open={open} className="w-[min(94vw,32rem)]">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-ink-soft">
          <BookMarked className="h-3.5 w-3.5 text-accent" />
          ashGPT · Getting started
        </div>

        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -12 }}
            transition={{ duration: 0.2 }}
          >
            <motion.div
              initial={{ scale: 0.9, rotate: -6 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: 'spring', stiffness: 260, damping: 18 }}
              className="mt-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-accent/10 text-accent"
            >
              <Current.Icon className="h-6 w-6" />
            </motion.div>
            <DialogTitle className="mt-4 font-serif text-2xl text-ink">
              {Current.title}
            </DialogTitle>
            <DialogDescription className="mt-2 text-sm text-ink-muted">
              {Current.body}
            </DialogDescription>
          </motion.div>
        </AnimatePresence>

        <div className="mt-6 flex items-center gap-1.5">
          {STEPS.map((_, i) => (
            <motion.span
              key={i}
              layout
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className={
                i === step
                  ? 'h-1.5 w-6 rounded-full bg-accent'
                  : 'h-1.5 w-1.5 rounded-full bg-parchment-warm'
              }
            />
          ))}
        </div>

        <div className="mt-6 flex items-center justify-between gap-2">
          <Button variant="ghost" size="sm" onClick={finish}>
            Skip tour
          </Button>
          <div className="flex gap-2">
            {step > 0 && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setStep((s) => Math.max(0, s - 1))}
              >
                Back
              </Button>
            )}
            {Current.cta && step !== STEPS.length - 1 ? (
              <Button size="sm" onClick={goToCta}>
                {Current.cta.label}
                <ArrowRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button size="sm" onClick={next}>
                {step === STEPS.length - 1 ? 'Done' : 'Next'}
                <ArrowRight className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
