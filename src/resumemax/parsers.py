"""Document parsers for resumes (PDF, DOCX, TXT) and job descriptions.

Each parser returns a normalized dict:
    {"raw_text": str, "sections": {section_name: section_text}}
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


SECTION_HEADERS = {
    "summary": [r"summary", r"profile", r"objective", r"about\s+me"],
    "experience": [r"experience", r"work\s+experience", r"employment", r"professional\s+experience"],
    "education": [r"education", r"academic\s+background"],
    "skills": [r"skills", r"technical\s+skills", r"core\s+competencies"],
    "projects": [r"projects", r"selected\s+projects"],
    "certifications": [r"certifications?", r"licenses?"],
}


def _read_pdf(path: Path) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_txt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def read_document(path: str | Path) -> str:
    """Read a document as raw text based on its extension."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(p)
    if suffix in {".docx", ".doc"}:
        return _read_docx(p)
    return _read_txt(p)


def split_sections(text: str) -> Dict[str, str]:
    """Split a resume into sections using header regexes.

    Builds one regex per canonical section, scans the document, and assigns
    the body between consecutive headers to the canonical section.
    """
    compiled: list[tuple[str, "re.Pattern[str]"]] = []
    for canonical, variants in SECTION_HEADERS.items():
        joined = "|".join(variants)
        compiled.append((canonical, re.compile(rf"^\s*(?:{joined})\s*:?\s*$", re.IGNORECASE | re.MULTILINE)))

    matches: list[tuple[int, int, str]] = []
    for canonical, rx in compiled:
        for m in rx.finditer(text):
            matches.append((m.start(), m.end(), canonical))
    matches.sort(key=lambda t: t[0])

    sections: Dict[str, str] = {}
    if not matches:
        sections["_preamble"] = text.strip()
        return sections

    sections["_preamble"] = text[: matches[0][0]].strip()
    for i, (start, end, canonical) in enumerate(matches):
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[end:body_end].strip()
        sections.setdefault(canonical, "")
        sections[canonical] = (sections[canonical] + "\n" + body).strip()
    return sections


def parse_resume(path: str | Path) -> dict:
    raw = read_document(path)
    return {"raw_text": raw, "sections": split_sections(raw), "source": str(path)}


def parse_job_description(text_or_path: str | Path) -> dict:
    s = str(text_or_path)
    p = Path(s) if len(s) < 1024 else None
    try:
        is_file = p is not None and p.exists() and p.is_file()
    except OSError:
        is_file = False
    raw = read_document(p) if is_file else s
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw) if s.strip()]
    return {"raw_text": raw, "sentences": sentences}
