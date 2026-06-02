"""TF-IDF keyword extraction and dictionary based skill extraction."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from .taxonomy import extract_skills_by_dict


def fit_tfidf(jd_corpus: Iterable[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, "any"]:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(list(jd_corpus))
    return vec, matrix


def top_terms(vec: TfidfVectorizer, doc: str, k: int = 30) -> List[Tuple[str, float]]:
    row = vec.transform([doc]).toarray()[0]
    terms = vec.get_feature_names_out()
    pairs = [(terms[i], float(row[i])) for i in row.nonzero()[0]]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:k]


def keyword_coverage(resume_text: str, jd_text: str, vec: TfidfVectorizer, k: int = 40) -> Dict:
    """Coverage = fraction of top-k JD terms that appear in the resume (case insensitive)."""
    top = top_terms(vec, jd_text, k=k)
    resume_l = resume_text.lower()
    hits = [t for t, _ in top if t in resume_l]
    misses = [t for t, _ in top if t not in resume_l]
    cov = len(hits) / len(top) if top else 0.0
    return {
        "coverage": cov,
        "top_terms": top,
        "hits": hits,
        "misses": misses,
    }


def skill_match(resume_text: str, jd_text: str) -> Dict:
    """Compare canonical skill sets extracted from the resume and JD."""
    r = set(extract_skills_by_dict(resume_text))
    j = set(extract_skills_by_dict(jd_text))
    overlap = r & j
    missing = j - r
    extra = r - j
    cov = len(overlap) / len(j) if j else 0.0
    return {
        "skill_coverage": cov,
        "resume_skills": sorted(r),
        "jd_skills": sorted(j),
        "overlap": sorted(overlap),
        "missing_in_resume": sorted(missing),
        "extra_in_resume": sorted(extra),
    }
