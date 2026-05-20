"""Case-citation extraction and verification helpers.

The synthesis node sometimes drifts: an LLM may import a familiar Property
Law case from its parametric memory even when that case is absent from the
retrieved sources. We catch that drift after synthesis by:

1. Extracting every plausible case citation from the final answer.
2. Checking each candidate against the concatenated retrieved chunk text.
3. Flagging citations that are not anchored in the sources so the
   verification node can ask the synthesis model to remove or hedge them.

The regex below is intentionally permissive (italics, underscores, plain
text) but constrained to legal-citation shapes (``X v Y``, ``X vs Y``) so
it does not match arbitrary "X v Y" prose.
"""

from __future__ import annotations

import re

# A case name token: capitalised word, possibly hyphenated, possibly with
# articles/connectors ("of", "the") between capitalised parts. Tightened to
# avoid matching ordinary sentences.
_CASE_NAME_PART = (
    r"[A-Z][A-Za-z'’-]+"
    r"(?:\s+(?:of|the|and|de|du|von|van)\s+[A-Z][A-Za-z'’-]+)?"
    r"(?:\s+[A-Z][A-Za-z'’-]+){0,4}"
)

# Match plain "X v Y", "X vs Y", and italicised variants (*X v Y* or _X v Y_).
# The 'v' separator may be `v`, `v.`, `vs`, `vs.` (case-insensitive).
_CASE_PATTERN = re.compile(
    r"(?P<wrap>[*_]{1,2})?"
    rf"(?P<left>{_CASE_NAME_PART})"
    r"\s+(?:v|vs)\.?\s+"
    rf"(?P<right>{_CASE_NAME_PART})"
    r"(?P=wrap)?",
    re.UNICODE,
)


# Tokens that should be trimmed if they appear at the start of a captured left
# side (e.g. "In Smith v Jones" → left="In Smith", trim leading "In" → "Smith").
# Without this, the regex's multi-word left (allowing "Buckinghamshire County
# Council") would absorb sentence-initial capitalised English words.
_LEFT_TRIM_PREFIXES = {
    "in",
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "section",
    "see",
    "of",
    "cited",
    "and",
    "or",
    "as",
    "by",
    "with",
    "to",
    "for",
    "at",
    "but",
    "from",
    "per",
    "however",
    "moreover",
    "thus",
}


def _trim_leading_stopwords(left: str) -> str:
    parts = left.split()
    while parts and parts[0].lower() in _LEFT_TRIM_PREFIXES:
        parts.pop(0)
    return " ".join(parts)


def extract_case_citations(text: str) -> list[str]:
    """Return unique, order-preserving list of case-citation strings from ``text``.

    Output entries are normalised to ``"<Left> v <Right>"`` form (lowercased
    separator, no italics wrappers, single spaces). Duplicates are dropped.
    """
    if not text:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for m in _CASE_PATTERN.finditer(text):
        left = re.sub(r"\s+", " ", m.group("left").strip())
        right = re.sub(r"\s+", " ", m.group("right").strip())
        left = _trim_leading_stopwords(left)
        if not left or not right:
            continue
        canonical = f"{left} v {right}"
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace + strip italic markers, for substring search."""
    cleaned = re.sub(r"[*_]", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower().strip()


def case_appears_in_sources(case: str, sources_text: str) -> bool:
    """Return True if either the full ``X v Y`` or both party tokens occur in sources."""
    norm_sources = _normalise(sources_text)
    norm_case = _normalise(case)
    if norm_case in norm_sources:
        return True
    parts = [p.strip() for p in re.split(r"\sv\s", norm_case) if p.strip()]
    if len(parts) != 2:
        return False
    left, right = parts
    return (left in norm_sources) and (right in norm_sources)


def find_unsupported_cases(answer: str, sources_text: str) -> list[str]:
    """Return citations in ``answer`` that do not appear in ``sources_text``."""
    candidates = extract_case_citations(answer)
    return [c for c in candidates if not case_appears_in_sources(c, sources_text)]
