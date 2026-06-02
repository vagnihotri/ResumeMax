"""ATS score aggregator combining keyword, semantic, and content layers."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

from .audit import audit_resume
from .embedding import semantic_score
from .keyword import keyword_coverage, skill_match
from .parsers import parse_job_description, parse_resume


DEFAULT_WEIGHTS = {"keyword": 0.35, "semantic": 0.45, "quality": 0.20}


@dataclass
class ATSResult:
    final_score: float
    sub_scores: Dict[str, float]
    keyword: dict
    skills: dict
    semantic: dict
    audit: dict
    recommendations: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fit_quick_tfidf(jd_text: str) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=2000, sublinear_tf=True
    )
    vec.fit([jd_text])
    return vec


def score(resume_path_or_text: str, jd_path_or_text: str, weights: Dict[str, float] | None = None) -> ATSResult:
    weights = weights or DEFAULT_WEIGHTS

    # Parse inputs.  `resume_path_or_text` may be either a filesystem path or
    # the raw text itself; long text strings can exceed the OS path length
    # limit so we guard the existence check.
    from pathlib import Path
    rp = Path(resume_path_or_text) if len(resume_path_or_text) < 1024 else None
    try:
        is_file = rp is not None and rp.exists() and rp.is_file()
    except OSError:
        is_file = False
    if is_file:
        resume = parse_resume(rp)
    else:
        resume = {"raw_text": resume_path_or_text, "sections": {}, "source": None}
    jd = parse_job_description(jd_path_or_text)

    r_text = resume["raw_text"]
    j_text = jd["raw_text"]

    # Keyword + skills
    vec = _fit_quick_tfidf(j_text)
    kw = keyword_coverage(r_text, j_text, vec, k=40)
    sk = skill_match(r_text, j_text)
    keyword_score = 0.5 * kw["coverage"] + 0.5 * sk["skill_coverage"]

    # Semantic
    sem = semantic_score(r_text, jd["sentences"])

    # Audit
    aud = audit_resume(r_text, source_path=resume.get("source"))

    sub = {
        "keyword": keyword_score,
        "semantic": sem["semantic_score"],
        "quality": aud["quality_score"],
    }
    final = 100.0 * (
        weights["keyword"] * sub["keyword"]
        + weights["semantic"] * sub["semantic"]
        + weights["quality"] * sub["quality"]
    )
    return ATSResult(
        final_score=round(final, 2),
        sub_scores={k: round(v, 4) for k, v in sub.items()},
        keyword=kw,
        skills=sk,
        semantic=sem,
        audit=aud,
    )
