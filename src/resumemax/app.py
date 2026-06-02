"""Streamlit UI for ResumeMax.

Run with:
    streamlit run src/resumemax/app.py
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from resumemax.parsers import read_document
from resumemax.recommend import attach_recommendations
from resumemax.score import score


st.set_page_config(page_title="ResumeMax", layout="wide")

st.title("ResumeMax")
st.caption("Automated Resume ATS Score Optimization via NLP and Semantic Embeddings")

# Session state for revision history
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {label, final_score, sub_scores}


# ---------- Demo data (Data Scientist - Retail Analytics) ----------
DEMO_JD = """Data Scientist - Retail Analytics

We are hiring a Data Scientist to drive insights for our retail merchandising team.

Responsibilities:
- Build forecasting and classification models in Python using scikit-learn and PyTorch.
- Run A/B tests and causal inference studies to evaluate pricing and promotion strategies.
- Develop SQL pipelines on Snowflake and visualize results in Tableau.
- Partner with merchandisers and engineers to ship recommendations into production.

Requirements:
- 3+ years of experience in data science or applied statistics.
- Strong programming skills in Python and SQL.
- Experience with scikit-learn, pandas, and at least one deep learning framework.
- Experience designing A/B tests and interpreting causal results.
- Familiarity with cloud data warehouses such as Snowflake, BigQuery, or Redshift.
- Excellent communication skills for non-technical stakeholders.
"""

DEMO_RESUME_WEAK = """Alex Taylor
Data Scientist - Retail Analytics | Remote | alex@example.com

SUMMARY
Data Scientist - Retail Analytics who is a team player and self-starter looking for an opportunity to leverage
my skills and add value in a fast paced environment.

EXPERIENCE
Data Scientist - Retail Analytics, Generic Corp (2022 - Present)
• Worked on various projects involving communication.
• Helped with deep learning when needed.
• Assisted senior team members with pandas.
• Worked on various projects involving python.

EDUCATION
B.S. Computer Science, State University, 2019

SKILLS
communication, deep learning, pandas, python
"""

DEMO_RESUME_STRONG = """Alex Taylor
Data Scientist - Retail Analytics | Remote | alex@example.com

SUMMARY
Data Scientist with 6 years of experience building forecasting and classification models for retail
merchandising. Delivered production ML systems in Python, scikit-learn, and PyTorch; designed A/B tests
and causal inference studies that informed pricing and promotion decisions.

EXPERIENCE
Senior Data Scientist, Acme Retail (2022 - Present)
• Built demand forecasting models in Python and scikit-learn that reduced stock-outs by 23% across 412 SKUs.
• Designed and analyzed 18 A/B tests on pricing and promotions, lifting gross margin by $4.2M annually.
• Shipped a PyTorch ranking model into the recommendations service, increasing click-through rate by 11%.
• Built SQL pipelines on Snowflake feeding Tableau dashboards used weekly by 40+ merchandisers.
• Partnered with engineering to deploy 6 models to production with full monitoring and rollback.

Data Scientist, Beta Commerce (2019 - 2022)
• Ran causal inference studies (difference-in-differences, synthetic control) to evaluate promo lift.
• Migrated legacy reporting from Redshift to Snowflake, cutting query latency by 64%.
• Mentored 4 junior analysts on Python, pandas, and experimental design.

EDUCATION
M.S. Statistics, State University, 2019
B.S. Mathematics, State University, 2017

SKILLS
Python, SQL, scikit-learn, PyTorch, pandas, Snowflake, BigQuery, Tableau, A/B testing, causal inference, communication
"""


def _load_demo(variant: str) -> None:
    st.session_state["resume_text"] = (
        DEMO_RESUME_STRONG if variant == "strong" else DEMO_RESUME_WEAK
    )
    st.session_state["jd_text"] = DEMO_JD


demo_c1, demo_c2, demo_c3 = st.columns([1, 1, 3])
demo_c1.button(
    "Load weak demo",
    on_click=_load_demo,
    args=("weak",),
    use_container_width=True,
    help="Pre-fills a weak Data Scientist resume (expect a low ATS score).",
)
demo_c2.button(
    "Load strong demo",
    on_click=_load_demo,
    args=("strong",),
    use_container_width=True,
    help="Pre-fills an optimized Data Scientist resume (expect a high ATS score).",
)
demo_c3.caption(
    "Demo: Data Scientist – Retail Analytics JD. Load either resume, then click **Score resume**."
)


# ---------- Sidebar: weights ----------
with st.sidebar:
    st.header("Scoring weights")
    w_kw = st.slider("Keyword", 0.0, 1.0, 0.35, 0.05)
    w_sem = st.slider("Semantic", 0.0, 1.0, 0.45, 0.05)
    w_q = st.slider("Quality", 0.0, 1.0, 0.20, 0.05)
    total = w_kw + w_sem + w_q
    if total > 0:
        weights = {"keyword": w_kw / total, "semantic": w_sem / total, "quality": w_q / total}
    else:
        weights = {"keyword": 0.35, "semantic": 0.45, "quality": 0.20}
    st.caption(f"Normalized: {weights}")

    if st.button("Clear revision history"):
        st.session_state.history = []
        st.rerun()


# ---------- Inputs ----------
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Resume")
    uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    resume_text = st.text_area(
        "or paste resume text",
        height=320,
        placeholder="Paste resume text here...",
        key="resume_text",
    )

with col_r:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste JD text",
        height=420,
        placeholder="Paste JD text here...",
        key="jd_text",
    )


def _resume_to_path(uploaded_file, fallback_text: str) -> tuple[str, str]:
    """Return (path_or_text, raw_text) for scoring."""
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower() or ".txt"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        return tmp.name, read_document(tmp.name)
    return fallback_text, fallback_text


# ---------- Score button ----------
if st.button("Score resume", type="primary", use_container_width=True):
    if (not uploaded and not resume_text.strip()) or not jd_text.strip():
        st.error("Provide both a resume (upload or paste) and a JD.")
    else:
        resume_arg, raw = _resume_to_path(uploaded, resume_text)
        with st.spinner("Scoring..."):
            result = attach_recommendations(score(resume_arg, jd_text, weights=weights))

        st.session_state.history.append({
            "label": f"v{len(st.session_state.history) + 1}",
            "final_score": result.final_score,
            "sub_scores": result.sub_scores,
        })

        # ----- Top metrics -----
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ATS Score", f"{result.final_score} / 100")
        m2.metric("Keyword", f"{result.sub_scores['keyword']:.2f}")
        m3.metric("Semantic", f"{result.sub_scores['semantic']:.2f}")
        m4.metric("Quality", f"{result.sub_scores['quality']:.2f}")

        # ----- Skill coverage -----
        st.subheader("Skill match")
        sk = result.skills
        c1, c2, c3 = st.columns(3)
        c1.markdown("**JD requires:**\n" + ("\n".join(f"- {s}" for s in sk["jd_skills"]) or "_(none)_"))
        c2.markdown("**Overlap:**\n" + ("\n".join(f"- [match] {s}" for s in sk["overlap"]) or "_(none)_"))
        c3.markdown("**Missing in resume:**\n" + ("\n".join(f"- [missing] {s}" for s in sk["missing_in_resume"]) or "_(none)_"))

        # ----- Audit findings -----
        st.subheader("Content audit")
        st.write(f"{result.audit['n_findings']} findings across {result.audit['n_bullets']} bullets")
        for f in result.audit["findings"][:25]:
            sev = f.get("severity", "low")
            tag = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}[sev]
            label = f.get("rule", "")
            body = f.get("bullet") or f.get("phrase") or ""
            st.markdown(f"`{tag}` **{label}** — {f.get('suggestion', '')}")
            if body:
                st.caption(f"context: {body[:200]}")

        # ----- Recommendations -----
        st.subheader("Top recommendations")
        for r in result.recommendations[:15]:
            st.markdown(f"- **[{r['priority']}] {r['type']}** — {r.get('advice', '')}")

        # ----- Per-requirement semantic table -----
        with st.expander("Per JD requirement semantic match"):
            for row in result.semantic["per_requirement"][:30]:
                st.markdown(f"- **sim={row['similarity']:.2f}** — JD: {row['jd_sentence'][:140]}")
                st.caption(f"   best resume bullet: {row['best_resume_sentence'][:160]}")

        # ----- JSON download -----
        st.download_button(
            "Download full JSON report",
            data=json.dumps(result.to_dict(), indent=2, default=str),
            file_name="resumemax_report.json",
            mime="application/json",
        )


# ---------- Revision history chart ----------
if st.session_state.history:
    st.subheader("Revision history")
    import pandas as pd
    df = pd.DataFrame([
        {
            "version": h["label"],
            "final": h["final_score"],
            "keyword": h["sub_scores"]["keyword"] * 100,
            "semantic": h["sub_scores"]["semantic"] * 100,
            "quality": h["sub_scores"]["quality"] * 100,
        }
        for h in st.session_state.history
    ])
    st.line_chart(df.set_index("version"))
    st.dataframe(df, use_container_width=True)
