/**
 * Approximate overlap highlighter for the citation popover.
 *
 * The model emits ``…some prose [S1]`` — we keep the last ~200 chars of
 * preceding prose on the chip (data-cite-context) and try to find the
 * longest n-gram of that prose that also appears in the cited chunk
 * snippet. If we find one, we wrap that span in <mark> inside the popover
 * so the reader can immediately see which part of the chunk grounds the
 * sentence. Falls back to plain snippet if no overlap above the floor.
 *
 * Cheap on purpose: word-level n-gram scan, no fuzzy matching, no stemming.
 */

const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'to', 'for', 'with',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'as', 'at', 'by',
  'this', 'that', 'these', 'those', 'it', 'its', 'from', 'but', 'not',
]);

const MIN_GRAM = 3;
const MAX_GRAM = 8;

function rawWords(text: string): string[] {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

function hasContent(words: string[]): boolean {
  return words.some((w) => !STOPWORDS.has(w));
}

/** Build a word-boundary regex out of consecutive context tokens. The
 *  separator ``[^a-z0-9]+`` matches whatever whitespace / punctuation
 *  appears between the words in the snippet, so the regex still hits when
 *  the snippet contains commas or extra spaces. */
function gramRegex(words: string[]): RegExp {
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return new RegExp(`\\b${escaped.join('[^a-z0-9]+')}\\b`, 'i');
}

/** Locate the best preceding-context n-gram inside ``snippet``. Returns
 *  character offsets into the snippet, or null if nothing meaningful
 *  overlaps. Word-level n-grams are tried from largest to smallest. */
export function findHighlightSpan(
  snippet: string,
  context: string,
): { start: number; end: number } | null {
  if (!snippet || !context) return null;
  const ctxWords = rawWords(context);
  if (ctxWords.length < MIN_GRAM) return null;

  for (let n = Math.min(MAX_GRAM, ctxWords.length); n >= MIN_GRAM; n--) {
    for (let i = 0; i + n <= ctxWords.length; i++) {
      const gramWords = ctxWords.slice(i, i + n);
      if (!hasContent(gramWords)) continue;
      const m = snippet.match(gramRegex(gramWords));
      if (m && m.index != null) {
        return { start: m.index, end: m.index + m[0].length };
      }
    }
  }
  return null;
}

export interface HighlightedSnippet {
  before: string;
  match: string;
  after: string;
}

/** Slice a snippet around the overlap span (if any) so the renderer can
 *  wrap just the matching span in <mark>. */
export function highlightSnippet(
  snippet: string,
  context: string,
): HighlightedSnippet | null {
  const span = findHighlightSpan(snippet, context);
  if (!span) return null;
  return {
    before: snippet.slice(0, span.start),
    match: snippet.slice(span.start, span.end),
    after: snippet.slice(span.end),
  };
}
