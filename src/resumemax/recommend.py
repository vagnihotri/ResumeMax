"""Recommendation engine that turns audit findings and gaps into actionable advice."""
from __future__ import annotations

from typing import Dict, List

from .score import ATSResult


STRONG_VERBS = [
    "led", "delivered", "designed", "built", "improved", "automated",
    "scaled", "shipped", "optimized", "architected", "launched",
]


def recommend(result: ATSResult, top_k_missing: int = 8) -> List[Dict]:
    recs: List[Dict] = []

    # 1) Missing skills from JD that the resume does not cover
    missing_skills = result.skills.get("missing_in_resume", [])[:top_k_missing]
    for s in missing_skills:
        recs.append({
            "type": "add_skill",
            "priority": "high",
            "skill": s,
            "advice": f"Add a bullet that demonstrates experience with {s} if you have it. "
                      f"For example: 'Used {s} to ...'",
        })

    # 2) Top JD keywords that the resume misses
    for term in result.keyword.get("misses", [])[:top_k_missing]:
        recs.append({
            "type": "add_keyword",
            "priority": "medium",
            "term": term,
            "advice": f"Consider naturally incorporating the JD term '{term}' if it matches your experience.",
        })

    # 3) Weak action verbs and missing metrics from the audit
    for f in result.audit.get("findings", []):
        if f["rule"] == "weak_action_verb":
            recs.append({
                "type": "rewrite_bullet",
                "priority": "medium",
                "bullet": f["bullet"],
                "advice": f"Rewrite starting with a strong verb such as: {', '.join(STRONG_VERBS[:6])}.",
            })
        elif f["rule"] == "missing_quantification":
            recs.append({
                "type": "quantify_bullet",
                "priority": "medium",
                "bullet": f["bullet"],
                "advice": "Add a number, percentage, dollar amount, or time saved to make the impact concrete.",
            })
        elif f["rule"] == "buzzword":
            recs.append({
                "type": "remove_buzzword",
                "priority": "low",
                "phrase": f.get("phrase", ""),
                "advice": f["suggestion"],
            })
        elif f["rule"] in {"multi_column_layout", "embedded_images", "very_short_resume"}:
            recs.append({
                "type": "fix_format",
                "priority": "high",
                "rule": f["rule"],
                "advice": f["suggestion"],
            })

    # 4) Low semantic match warning
    if result.sub_scores["semantic"] < 0.35:
        recs.append({
            "type": "improve_semantic_alignment",
            "priority": "high",
            "advice": "Several JD requirements have no closely matching resume content. "
                      "Add bullets that paraphrase the JD requirements with concrete examples from your work.",
        })

    return recs


def attach_recommendations(result: ATSResult) -> ATSResult:
    result.recommendations = recommend(result)
    return result
