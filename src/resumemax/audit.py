"""Rule based content audit: action verbs, quantifiable metrics, buzzwords, formatting."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "lexicons" / "action_verbs.json"

_QUANT_RE = re.compile(
    r"(\d+\s?%|\$\s?\d+|\d+[\d,\.]*\s?(k|m|b|million|billion|users|customers|requests|hours|days|months|years|x))",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[•\-\*\u2022]\s*(.+)$", re.MULTILINE)


@lru_cache(maxsize=1)
def _lex() -> Dict[str, List[str]]:
    return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))


def extract_bullets(text: str) -> List[str]:
    bullets = [m.group(1).strip() for m in _BULLET_RE.finditer(text)]
    if bullets:
        return bullets
    # Fall back to non-empty lines that look like accomplishments
    return [ln.strip() for ln in text.splitlines() if 20 < len(ln.strip()) < 400]


def find_weak_verbs(bullets: List[str]) -> List[Dict]:
    weak = _lex()["weak"]
    findings = []
    for b in bullets:
        bl = b.lower()
        for phrase in weak:
            if bl.startswith(phrase) or f" {phrase} " in bl[:60]:
                findings.append({
                    "rule": "weak_action_verb",
                    "severity": "medium",
                    "bullet": b,
                    "phrase": phrase,
                    "suggestion": "Replace with a stronger action verb such as led, delivered, designed, built, or improved.",
                })
                break
    return findings


def find_missing_metrics(bullets: List[str]) -> List[Dict]:
    findings = []
    for b in bullets:
        if not _QUANT_RE.search(b):
            findings.append({
                "rule": "missing_quantification",
                "severity": "medium",
                "bullet": b,
                "suggestion": "Quantify this bullet with a number, percentage, or dollar amount where possible.",
            })
    return findings


def find_buzzwords(text: str) -> List[Dict]:
    findings = []
    tl = text.lower()
    for w in _lex()["buzzwords"]:
        if w in tl:
            findings.append({
                "rule": "buzzword",
                "severity": "low",
                "phrase": w,
                "suggestion": f"Remove or rephrase the buzzword '{w}' with a concrete accomplishment.",
            })
    return findings


def find_formatting_issues(text: str, source_path: str | None = None) -> List[Dict]:
    findings = []
    if source_path and source_path.lower().endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(source_path) as pdf:
                page = pdf.pages[0]
                # Heuristic: detect multi-column layouts via x0 clusters
                xs = sorted({round(w["x0"] / 50) * 50 for w in page.extract_words()})
                if len(xs) >= 3 and (max(xs) - min(xs)) > 200:
                    findings.append({
                        "rule": "multi_column_layout",
                        "severity": "high",
                        "suggestion": "Use a single column layout. Multi-column resumes often confuse ATS parsers.",
                    })
                if len(page.images) > 0:
                    findings.append({
                        "rule": "embedded_images",
                        "severity": "medium",
                        "suggestion": "Avoid embedded images and icons. ATS systems cannot read text from images.",
                    })
        except Exception:
            pass
    if len(text) < 400:
        findings.append({
            "rule": "very_short_resume",
            "severity": "high",
            "suggestion": "Resume text appears very short. Make sure all sections are present and parseable.",
        })
    return findings


def audit_resume(text: str, source_path: str | None = None) -> Dict:
    bullets = extract_bullets(text)
    findings = (
        find_weak_verbs(bullets)
        + find_missing_metrics(bullets)
        + find_buzzwords(text)
        + find_formatting_issues(text, source_path)
    )
    # Quality score: 1.0 minus penalty per finding, capped.
    sev_weight = {"low": 0.01, "medium": 0.03, "high": 0.06}
    penalty = sum(sev_weight.get(f.get("severity", "low"), 0.01) for f in findings)
    quality = max(0.0, 1.0 - penalty)
    return {
        "quality_score": quality,
        "n_findings": len(findings),
        "findings": findings,
        "n_bullets": len(bullets),
    }
