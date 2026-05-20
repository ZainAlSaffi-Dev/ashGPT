"""Per-mime text extractors.

Each ``extract_*`` returns a list of ``ExtractedSection``:

  * ``text``  — non-empty extracted text
  * ``page``  — 1-based page number for paginated formats (PDF), else None
  * ``meta``  — any extractor-specific extras (e.g. image_path for slides)

The pipeline (``src.ingestion.pipeline``) decides how to chunk each section.
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    text: str
    page: int | None = None
    meta: dict = field(default_factory=dict)


# ── PDF ───────────────────────────────────────────────────────────────────────


def extract_pdf(path: str | Path) -> list[ExtractedSection]:
    """Return one section per non-empty PDF page."""
    import fitz  # type: ignore

    path = Path(path)
    doc = fitz.open(path)
    out: list[ExtractedSection] = []
    try:
        for i, page in enumerate(doc, start=1):
            text = (page.get_text() or "").strip()
            if text:
                out.append(ExtractedSection(text=text, page=i))
    finally:
        doc.close()
    return out


# ── DOCX ──────────────────────────────────────────────────────────────────────


def extract_docx(path: str | Path) -> list[ExtractedSection]:
    """Return one section containing the full doc's paragraph text.

    DOCX has no native pages — paginate later only if needed. Headings are
    preserved as plain text so the chunker keeps natural section breaks.
    """
    import docx  # python-docx

    document = docx.Document(str(path))
    parts: list[str] = []
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    text = "\n\n".join(parts)
    return [ExtractedSection(text=text)] if text else []


# ── Plain text + Markdown ─────────────────────────────────────────────────────


def extract_text(path: str | Path) -> list[ExtractedSection]:
    """Read a UTF-8 .txt or .md file as a single section."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="latin-1")
    text = text.strip()
    return [ExtractedSection(text=text)] if text else []


# ── Images (VLM) ──────────────────────────────────────────────────────────────


_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _describe_image(image_path: Path) -> str:
    """Send an image to the Gemini VLM and return its description."""
    from google import genai
    from google.genai import types

    from src.config import SLIDE_DESCRIPTION_PROMPT, VLM_MODEL

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
    client = genai.Client()
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=[
            SLIDE_DESCRIPTION_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ],
    )
    return response.text or ""


def extract_image(path: str | Path) -> list[ExtractedSection]:
    """Run the Gemini VLM to describe an image. Returns one section.

    The original image path is stashed on ``meta.image_path`` so the agent can
    cite the source image in the UI.
    """
    description = _describe_image(Path(path)).strip()
    if not description:
        return []
    return [
        ExtractedSection(
            text=description,
            meta={"image_path": str(path), "doc_type_hint": "image"},
        )
    ]


# ── Mime dispatch ─────────────────────────────────────────────────────────────


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def detect_mime(path: str | Path, fallback: str | None = None) -> str:
    """Return a best-effort MIME for ``path``.

    Falls back to ``fallback`` (the upload-time client-reported MIME), else
    ``application/octet-stream``.
    """
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    suffix = Path(path).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".docx": _DOCX_MIME,
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, fallback or "application/octet-stream")


def extract(path: str | Path, mime: str | None = None) -> list[ExtractedSection]:
    """Dispatch on MIME (or extension) to the right extractor."""
    m = mime or detect_mime(path)
    if m == "application/pdf":
        return extract_pdf(path)
    if m == _DOCX_MIME:
        return extract_docx(path)
    if m in ("text/plain", "text/markdown"):
        return extract_text(path)
    if m.startswith("image/"):
        return extract_image(path)
    raise ValueError(f"unsupported mime: {m}")
