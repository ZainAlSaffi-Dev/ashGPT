import { describe, expect, it, vi } from 'vitest';

import { dispatch, parseSSEStream } from './streaming';

describe('dispatch', () => {
  it('parses JSON node events', () => {
    const onNode = vi.fn();
    dispatch('node', '{"node":"retrieval"}', { onNode });
    expect(onNode).toHaveBeenCalledWith('retrieval');
  });

  it('passes sources array through', () => {
    const onSources = vi.fn();
    const payload = {
      sources: [{ source: 'notes.pdf', doc_type: 'note', week: 'week_3', snippet: 'x' }],
    };
    dispatch('sources', JSON.stringify(payload), { onSources });
    expect(onSources).toHaveBeenCalledWith(payload.sources);
  });

  it('coerces done payload fields', () => {
    const onDone = vi.fn();
    dispatch(
      'done',
      JSON.stringify({
        session_id: 'abc',
        intent: 'ratio',
        latency_ms: 1234,
        final_answer: 'Adverse possession is...',
      }),
      { onDone },
    );
    expect(onDone).toHaveBeenCalledWith({
      session_id: 'abc',
      intent: 'ratio',
      latency_ms: 1234,
      final_answer: 'Adverse possession is...',
    });
  });

  it('surfaces errors with default message when detail missing', () => {
    const onError = vi.fn();
    dispatch('error', '{}', { onError });
    expect(onError).toHaveBeenCalledWith('unknown error');
  });

  it('ignores unknown event names without throwing', () => {
    expect(() => dispatch('mystery', '{}', {})).not.toThrow();
  });

  it('handles non-JSON data gracefully', () => {
    const onAnswerChunk = vi.fn();
    dispatch('answer_chunk', 'not json {{', { onAnswerChunk });
    // text field missing → coerced to empty
    expect(onAnswerChunk).toHaveBeenCalledWith('');
  });
});

describe('parseSSEStream', () => {
  it('emits node + answer_chunk + done in order', async () => {
    const chunks = [
      'event: node\ndata: {"node":"router"}\n\n',
      'event: node\ndata: {"node":"retrieval"}\n\n',
      'event: answer_chunk\ndata: {"text":"Adverse possession is..."}\n\n',
      'event: done\ndata: {"session_id":"s1","intent":"ratio","latency_ms":42,"final_answer":"Adverse possession is..."}\n\n',
    ];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const enc = new TextEncoder();
        for (const c of chunks) controller.enqueue(enc.encode(c));
        controller.close();
      },
    });

    const seen: string[] = [];
    await parseSSEStream(stream, {
      onNode: (n) => seen.push(`node:${n}`),
      onAnswerChunk: (t) => seen.push(`chunk:${t}`),
      onDone: (d) => seen.push(`done:${d.session_id}:${d.intent}`),
    });
    expect(seen).toEqual([
      'node:router',
      'node:retrieval',
      'chunk:Adverse possession is...',
      'done:s1:ratio',
    ]);
  });
});
