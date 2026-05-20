'use client';

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';

import { generateExam, submitExam } from '@/lib/api';
import type { ExamPayload, ExamResult } from '@/lib/types';
import { cn } from '@/lib/utils';

type Scope = 'all' | 'week' | 'past_paper';

export default function ExamPage() {
  const { getToken } = useAuth();
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

  const onGenerate = async () => {
    setBusy(true);
    setError(null);
    setExam(null);
    setResult(null);
    setAnswers({});
    try {
      const token = (await getToken()) ?? undefined;
      const payload = await generateExam({
        scope_type: scopeType,
        scope_value: scopeValue || undefined,
        num_mcq: numMcq,
        num_short: numShort,
        difficulty,
      }, token);
      setExam(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
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
        Generate multiple-choice and short-answer questions from your notes or
        upload a past paper to grade your attempt.
      </p>

      {!exam && (
        <div className="mt-6 space-y-4 rounded-lg border border-parchment-warm bg-parchment p-6">
          <div className="grid grid-cols-2 gap-4">
            <label className="text-sm text-ink">
              Scope
              <select
                value={scopeType}
                onChange={(e) => setScopeType(e.target.value as Scope)}
                className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
              >
                <option value="all">All my notes</option>
                <option value="week">A specific week</option>
                <option value="past_paper">A past paper</option>
              </select>
            </label>
            {scopeType !== 'all' && (
              <label className="text-sm text-ink">
                {scopeType === 'week' ? 'Week' : 'Past paper file id'}
                <input
                  value={scopeValue}
                  onChange={(e) => setScopeValue(e.target.value)}
                  placeholder={scopeType === 'week' ? 'week_3' : 'file id'}
                  className="mt-1 w-full rounded border border-parchment-warm bg-parchment px-2 py-1"
                />
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
                onChange={(e) => setDifficulty(e.target.value as 'easy' | 'medium' | 'hard')}
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
            disabled={busy}
            className={cn(
              'w-full rounded-md bg-accent px-4 py-2 text-parchment shadow',
              busy ? 'opacity-50' : 'hover:bg-accent-hover',
            )}
          >
            {busy ? 'Generating…' : 'Generate exam'}
          </button>
          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
      )}

      {exam && (
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
