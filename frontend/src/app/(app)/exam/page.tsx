'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@clerk/nextjs';
import { FileStack } from 'lucide-react';

import { OnboardingChecklist } from '@/components/OnboardingChecklist';
import { generateExam, submitExam } from '@/lib/api';
import { useFiles, useOnboarding } from '@/lib/queries';
import type { ExamPayload, ExamResult } from '@/lib/types';
import { cn } from '@/lib/utils';

type Scope = 'all' | 'file' | 'past_paper';

export default function ExamPage() {
  const { getToken } = useAuth();
  const filesQuery = useFiles();
  const onboarding = useOnboarding();

  const [scopeType, setScopeType] = useState<Scope>('all');
  const [scopeValue, setScopeValue] = useState('');
  const [numMcq, setNumMcq] = useState(5);
  const [numShort, setNumShort] = useState(2);
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [exam, setExam] = useState<ExamPayload | null>(null);
  const [answers, setAnswers] = useState<Record<string, number | string>>({});
  const [result, setResult] = useState<ExamResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const readyFiles = (filesQuery.data ?? []).filter((f) => f.status === 'ready');
  const hasReadyFiles = readyFiles.length > 0;

  const onGenerate = async () => {
    setBusy(true);
    setError(null);
    setExam(null);
    setResult(null);
    setAnswers({});
    try {
      const token = (await getToken()) ?? undefined;
      const payload = await generateExam(
        {
          scope_type: scopeType,
          scope_value: scopeValue || undefined,
          num_mcq: numMcq,
          num_short: numShort,
          difficulty,
        },
        token,
      );
      setExam(payload);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // Backend returns 409 with "no chunks available" when the namespace is
      // empty — show a friendlier prompt instead of the raw API string.
      if (msg.toLowerCase().includes('no chunks')) {
        setError(
          'No indexed material yet. Add at least one document to your library before generating an exam.',
        );
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = async () => {
    if (!exam) return;
    setBusy(true);
    try {
      const token = (await getToken()) ?? undefined;
      const r = await submitExam(exam.id, answers, token);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <h1 className="font-serif text-2xl text-ink">Exam practice</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Generate MCQ + short-answer questions grounded in the files in your library,
        then grade your attempt against the rubric.
      </p>

      {!onboarding.isComplete && (
        <div className="mt-6">
          <OnboardingChecklist variant="full" />
        </div>
      )}

      {!hasReadyFiles ? (
        <EmptyLibraryCallout filesLoading={filesQuery.isLoading} />
      ) : !exam ? (
        <div className="mt-6 space-y-4 rounded-lg border border-parchment-warm bg-parchment p-6">
          <div className="grid grid-cols-2 gap-4">
            <label className="text-sm text-ink">
              Scope
              <select
                value={scopeType}
                onChange={(e) => {
                  setScopeType(e.target.value as Scope);
                  setScopeValue('');
                }}
                className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
              >
                <option value="all">All my files</option>
                <option value="file">A specific file</option>
                <option value="past_paper">Past papers only</option>
              </select>
            </label>
            {scopeType === 'file' && (
              <label className="text-sm text-ink">
                File
                <select
                  value={scopeValue}
                  onChange={(e) => setScopeValue(e.target.value)}
                  className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
                >
                  <option value="">— pick a file —</option>
                  {readyFiles.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="text-sm text-ink">
              MCQs
              <input
                type="number"
                min={0}
                max={20}
                value={numMcq}
                onChange={(e) => setNumMcq(Number(e.target.value))}
                className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
              />
            </label>
            <label className="text-sm text-ink">
              Short-answer
              <input
                type="number"
                min={0}
                max={10}
                value={numShort}
                onChange={(e) => setNumShort(Number(e.target.value))}
                className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
              />
            </label>
            <label className="col-span-2 text-sm text-ink">
              Difficulty
              <select
                value={difficulty}
                onChange={(e) =>
                  setDifficulty(e.target.value as 'easy' | 'medium' | 'hard')
                }
                className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </label>
          </div>
          <button
            onClick={onGenerate}
            disabled={busy || (scopeType === 'file' && !scopeValue)}
            className={cn(
              'w-full rounded-md bg-accent px-4 py-2 text-parchment shadow',
              busy || (scopeType === 'file' && !scopeValue)
                ? 'cursor-not-allowed opacity-50'
                : 'hover:bg-accent-hover',
            )}
          >
            {busy ? 'Generating…' : 'Generate exam'}
          </button>
          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
      ) : (
        <div className="mt-6 space-y-6">
          {exam.mcq.map((q, i) => {
            const qid = `mcq_${i}`;
            return (
              <div key={qid} className="rounded-lg border border-parchment-warm bg-parchment p-4">
                <p className="font-medium text-ink">
                  {i + 1}. {q.question}
                </p>
                <div className="mt-3 space-y-2">
                  {q.options.map((opt, oi) => (
                    <label key={oi} className="flex items-center gap-2 text-sm text-ink">
                      <input
                        type="radio"
                        name={qid}
                        checked={answers[qid] === oi}
                        onChange={() => setAnswers({ ...answers, [qid]: oi })}
                      />
                      {opt}
                    </label>
                  ))}
                </div>
                {result?.per_question.find((r) => r.question_id === qid) && (
                  <p className="mt-2 text-xs text-ink-muted">
                    {result.per_question.find((r) => r.question_id === qid)?.feedback}
                  </p>
                )}
              </div>
            );
          })}
          {exam.short.map((q, i) => {
            const qid = `short_${i}`;
            return (
              <div key={qid} className="rounded-lg border border-parchment-warm bg-parchment p-4">
                <p className="font-medium text-ink">
                  {exam.mcq.length + i + 1}. {q.question}
                </p>
                <textarea
                  rows={5}
                  value={(answers[qid] as string) ?? ''}
                  onChange={(e) => setAnswers({ ...answers, [qid]: e.target.value })}
                  className="mt-3 w-full rounded border border-parchment-warm bg-parchment px-3 py-2 text-sm"
                />
                {result?.per_question.find((r) => r.question_id === qid) && (
                  <p className="mt-2 text-xs text-ink-muted">
                    {result.per_question.find((r) => r.question_id === qid)?.feedback}
                  </p>
                )}
              </div>
            );
          })}

          {!result && (
            <button
              onClick={onSubmit}
              disabled={busy}
              className="w-full rounded-md bg-accent px-4 py-2 text-parchment shadow hover:bg-accent-hover"
            >
              {busy ? 'Grading…' : 'Submit for grading'}
            </button>
          )}

          {result && (
            <div className="rounded-lg border border-accent bg-parchment-warm/40 p-4">
              <p className="font-serif text-xl text-ink">Score: {result.score.toFixed(1)}</p>
              <button
                onClick={() => {
                  setExam(null);
                  setResult(null);
                  setAnswers({});
                }}
                className="mt-3 text-sm text-accent hover:underline"
              >
                Start another
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EmptyLibraryCallout({ filesLoading }: { filesLoading: boolean }) {
  if (filesLoading) {
    return <p className="mt-6 text-sm text-ink-muted">Loading your library…</p>;
  }
  return (
    <div className="mt-6 flex flex-col items-center gap-3 rounded-2xl border border-dashed border-parchment-warm bg-parchment-warm/30 p-10 text-center">
      <FileStack className="h-10 w-10 text-accent" />
      <h2 className="font-serif text-lg text-ink">Add a file before practising</h2>
      <p className="max-w-md text-sm text-ink-muted">
        Exam questions are grounded in the documents in your library. Drop a PDF
        or paste in notes first — once at least one file is indexed, the
        generator unlocks.
      </p>
      <Link
        href="/library"
        className="mt-2 rounded-md bg-accent px-4 py-2 text-sm text-parchment shadow transition hover:bg-accent-hover"
      >
        Add files →
      </Link>
    </div>
  );
}
