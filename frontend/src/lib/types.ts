// Wire types — must mirror backend/api/schemas.py.

export type Intent = 'ratio' | 'chronology' | 'summary' | 'general';

export interface SourceHit {
  chunk_id?: string | null;
  file_id?: string | null;
  file_name?: string | null;
  project_id?: string | null;
  folder_id?: string | null;
  page?: number | null;
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
    scope?: RetrievalScope | null;
    intent: Intent | null;
    latency_ms: number;
    final_answer: string;
  };
  error?: { detail: string };
}

export interface SessionSummary {
  id: string;
  title: string;
  project_id?: string | null;
  folder_id?: string | null;
  scope?: RetrievalScope | null;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  scope?: RetrievalScope | null;
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
  project_id?: string | null;
  folder_id?: string | null;
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

export interface Project {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  color?: string | null;
  archived_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface Folder {
  id: string;
  project_id: string;
  parent_id?: string | null;
  name: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export type RetrievalScope =
  | { type: 'all' }
  | { type: 'project'; project_id: string }
  | { type: 'folder'; project_id?: string | null; folder_id: string }
  | { type: 'files'; project_id?: string | null; folder_id?: string | null; file_ids: string[] }
  | { type: 'week'; project_id?: string | null; folder_id?: string | null; week: string }
  | { type: 'doc_type'; project_id?: string | null; folder_id?: string | null; doc_types: string[] };

export interface FileListScope {
  projectId?: string | null;
  folderId?: string | null;
  status?: string | null;
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
