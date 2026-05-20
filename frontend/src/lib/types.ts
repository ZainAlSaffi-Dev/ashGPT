// Wire types — must mirror backend/api/schemas.py.

export type Intent = 'ratio' | 'chronology' | 'summary' | 'general';

export interface SourceHit {
  source: string | null;
  doc_type: string | null;
  week: string | null;
  snippet: string;
}

export interface VerificationReport {
  all_supported?: boolean;
  unsupported_citations?: string[];
  details?: Record<string, unknown>;
}

export interface ChatHistoryOverflow {
  dropped_turns: number;
  truncated_messages: number;
}

export interface ChatStreamEvents {
  node?: { node: string; session_id?: string };
  sources?: { sources: SourceHit[] };
  irac?: { irac: string };
  mermaid?: { diagram: string };
  verification?: { report: VerificationReport };
  history_overflow?: ChatHistoryOverflow;
  answer_chunk?: { text: string };
  done?: {
    session_id: string;
    intent: Intent | null;
    latency_ms: number;
    final_answer: string;
  };
  error?: { detail: string };
}

export interface SessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  intent?: string | null;
  retrieved_chunk_ids?: string[] | null;
  sources?: SourceHit[] | null;
  irac?: string | null;
  mermaid?: string | null;
  latency_ms?: number | null;
  verification?: VerificationReport | null;
  created_at: string;
}

export interface FileMeta {
  id: string;
  name: string;
  mime: string;
  size_bytes: number;
  status: 'uploaded' | 'processing' | 'queued' | 'ready' | 'failed';
  error?: string | null;
  // Free-form so callers can categorise (case, statute, note, past_paper,
  // transcript, slide, …) without the frontend imposing a fixed taxonomy.
  doc_type: string;
  week: string | null;
  chunk_count: number;
  created_at: string;
}

export interface PresignResponse {
  file_id: string;
  upload_url: string;
  blob_key: string;
  method: 'PUT' | 'POST';
}

export interface ExamMCQ {
  question: string;
  options: string[];
  correct_idx: number;
  explanation?: string;
  source_chunks?: string[];
}

export interface ExamShortAnswer {
  question: string;
  model_answer: string;
  grading_rubric: string[];
  source_chunks?: string[];
}

export interface ExamPayload {
  id: string;
  mcq: ExamMCQ[];
  short: ExamShortAnswer[];
}

export interface ExamResult {
  score: number;
  per_question: Array<{
    question_id: string;
    score: number;
    feedback?: string;
    rubric_hits?: string[];
  }>;
}
