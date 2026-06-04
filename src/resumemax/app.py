"""Streamlit UI for ResumeMax.

Run with:
    streamlit run src/resumemax/app.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

# Make the package importable when this script is launched directly (e.g. by
# Streamlit Community Cloud) without PYTHONPATH=src being set. The package
# lives at <repo>/src/resumemax/, so add <repo>/src/ to sys.path.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

from resumemax.parsers import read_document
from resumemax.recommend import attach_recommendations
from resumemax.score import score


st.set_page_config(page_title="ResumeMax", layout="wide")

st.title("ResumeMax")
st.caption(
    "Automated Resume ATS Score Optimization via NLP and Semantic Embeddings. "
    "Paste **or upload** both the resume and the job description, every text "
    "box is fully editable, and the demo buttons are just convenient starting "
    "points."
)

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
    help="Pre-fills a weak Data Scientist resume + matching JD (expect a low Resume Score). You can then edit either box freely.",
)
demo_c2.button(
    "Load strong demo",
    on_click=_load_demo,
    args=("strong",),
    use_container_width=True,
    help="Pre-fills an optimized Data Scientist resume + matching JD (expect a high Resume Score). You can then edit either box freely.",
)
demo_c3.caption(
    "Demos are optional starter content. You can paste **any** resume and "
    "**any** JD, or upload them as PDF / DOCX / TXT, and re-score."
)


# ---------- Sidebar: weights ----------
with st.sidebar:
    st.header("Scoring weights")
    st.caption(
        "ResumeMax combines three signals into a single 0–100 **Resume Score**. "
        "Slide to change how much each one contributes, values are normalized "
        "to sum to 1 automatically."
    )
    w_kw = st.slider("Keyword weight", 0.0, 1.0, 0.35, 0.05,
                     help="Weight on TF-IDF keyword + skill-taxonomy overlap.")
    w_sem = st.slider("Semantic weight", 0.0, 1.0, 0.45, 0.05,
                      help="Weight on Sentence-BERT cosine similarity between JD requirements and resume bullets.")
    w_q = st.slider("Quality weight", 0.0, 1.0, 0.20, 0.05,
                    help="Weight on content-audit findings (weak verbs, missing metrics, buzzwords, formatting).")
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
    uploaded_resume = st.file_uploader(
        "Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], key="resume_upload"
    )
    resume_text = st.text_area(
        "…or paste resume text (editable)",
        height=320,
        placeholder="Paste resume text here, or upload a file above.",
        key="resume_text",
    )

with col_r:
    st.subheader("Job Description")
    uploaded_jd = st.file_uploader(
        "Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], key="jd_upload"
    )
    jd_text = st.text_area(
        "…or paste JD text (editable)",
        height=320,
        placeholder="Paste the JD here, or upload a file above. You can replace this at any time and re-score.",
        key="jd_text",
    )


def _doc_to_path_and_text(uploaded_file, fallback_text: str) -> tuple[str, str]:
    """Return (path_or_text, raw_text). File uploads beat the textarea."""
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
    have_resume = (uploaded_resume is not None) or bool(resume_text.strip())
    have_jd = (uploaded_jd is not None) or bool(jd_text.strip())
    if not have_resume or not have_jd:
        st.error(
            "Provide **both** a resume (upload or paste) and a job description "
            "(upload or paste)."
        )
    else:
        resume_arg, _ = _doc_to_path_and_text(uploaded_resume, resume_text)
        jd_arg, _ = _doc_to_path_and_text(uploaded_jd, jd_text)
        with st.spinner("Scoring..."):
            result = attach_recommendations(score(resume_arg, jd_arg, weights=weights))

        st.session_state.history.append({
            "label": f"v{len(st.session_state.history) + 1}",
            "final_score": result.final_score,
            "sub_scores": result.sub_scores,
        })

        # ----- Top metrics with inline explanations -----
        st.subheader("Your Resume Score")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Resume Score (0–100)",
            f"{result.final_score}",
            help=(
                "Weighted sum of the three components below. This is a "
                "ResumeMax-internal score that estimates ATS keyword match, "
                "semantic alignment, and writing quality, it is not a "
                "prediction of employability or interview odds."
            ),
        )
        m2.metric(
            "Keyword (0–1)",
            f"{result.sub_scores['keyword']:.2f}",
            help=(
                "Average of (a) fraction of top-40 TF-IDF JD terms present in "
                "the resume and (b) fraction of taxonomy skills in the JD that "
                "also appear in the resume. 1.0 = every JD term and skill is "
                "present somewhere in the resume."
            ),
        )
        m3.metric(
            "Semantic (0–1)",
            f"{result.sub_scores['semantic']:.2f}",
            help=(
                "Mean of, for each JD sentence, the cosine similarity to its "
                "best-matching resume sentence (Sentence-BERT MiniLM-L6-v2). "
                "1.0 = every JD requirement has a near-paraphrase in the resume."
            ),
        )
        m4.metric(
            "Quality (0–1)",
            f"{result.sub_scores['quality']:.2f}",
            help=(
                "1.0 minus a penalty per content-audit finding: weak action "
                "verbs (−0.03 each), missing quantification (−0.03), buzzwords "
                "(−0.01), formatting issues such as multi-column / images / "
                "very short text (−0.06)."
            ),
        )

        # ----- Score breakdown: how the 0-100 was built -----
        st.markdown("#### How your Resume Score was calculated")
        st.caption(
            "Each component score (0–1) is multiplied by its weight, then by "
            "100, and the three contributions are summed. The numbers below add "
            "up to your Resume Score, this is the **only** computation behind "
            "the headline number."
        )
        breakdown_rows = []
        for comp in ("keyword", "semantic", "quality"):
            sub = float(result.sub_scores.get(comp, 0.0))
            w = weights[comp]
            pts = 100.0 * w * sub
            possible = 100.0 * w
            breakdown_rows.append({
                "Component": comp,
                "Sub-score (0–1)": round(sub, 3),
                "Weight": round(w, 3),
                "Points contributed": round(pts, 2),
                "Points possible": round(possible, 2),
                "Headroom": round(possible - pts, 2),
            })
        import pandas as pd  # noqa: WPS433  (local import keeps cold-start light)
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)
        biggest = max(breakdown_rows, key=lambda r: r["Headroom"])
        st.info(
            f"Biggest lever: **{biggest['Component']}** "
            f"(headroom = {biggest['Headroom']} pts). Improvements to this "
            "component will move your Resume Score the most."
        )

        # ----- Skill coverage -----
        st.subheader("Skill match")
        st.caption(
            "Skills are extracted from both documents using a curated "
            "taxonomy of 35 canonical entries: 30 hard skills (programming "
            "languages such as Python, Java, JavaScript, TypeScript, SQL, "
            "C++, R, Scala, Go; cloud platforms AWS, Azure, GCP; data and "
            "ML frameworks scikit-learn, PyTorch, TensorFlow, pandas, NumPy, "
            "Spark, Hadoop; ops and tooling Docker, Kubernetes, Git, REST "
            "API, GraphQL; BI and analytics Tableau, Power BI, Excel; and "
            "domain areas NLP, machine learning, deep learning) plus 5 soft "
            "skills (communication, leadership, teamwork, problem solving, "
            "project management). Each canonical skill carries an alias list "
            "(e.g. 'k8s' for Kubernetes, 'sklearn' for scikit-learn, 'js' "
            "for JavaScript), giving 84 surface forms in total, and matching "
            "is case-insensitive. 'Missing in resume' below is the set "
            "of canonical skills found in the JD but not in the resume, and "
            "it drives the **add_skill** recommendations."
        )
        sk = result.skills
        c1, c2, c3 = st.columns(3)
        c1.markdown("**JD requires:**\n" + ("\n".join(f"- {s}" for s in sk["jd_skills"]) or "_(none)_"))
        c2.markdown("**Overlap (present in resume):**\n" + ("\n".join(f"- {s}" for s in sk["overlap"]) or "_(none)_"))
        c3.markdown("**Missing in resume:**\n" + ("\n".join(f"- {s}" for s in sk["missing_in_resume"]) or "_(none)_"))

        # ----- Audit findings -----
        st.subheader("Content audit")
        st.caption(
            "Rule-based checks on every bullet. Each finding costs a small "
            "amount from the **Quality** sub-score, which in turn reduces your "
            "Resume Score in proportion to the Quality weight."
        )
        st.write(f"{result.audit['n_findings']} findings across {result.audit['n_bullets']} bullets")
        for f in result.audit["findings"][:25]:
            sev = f.get("severity", "low")
            tag = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}[sev]
            label = f.get("rule", "")
            body = f.get("bullet") or f.get("phrase") or ""
            st.markdown(f"`{tag}` **{label}**, {f.get('suggestion', '')}")
            if body:
                st.caption(f"context: {body[:200]}")

        # ----- Recommendations -----
        st.subheader("Top recommendations")
        st.caption(
            "Suggested edits to raise your **Resume Score**. The score "
            "measures keyword / semantic / quality alignment with the JD, it "
            "is not a measure of employability or hiring outcomes."
        )
        for r in result.recommendations[:15]:
            st.markdown(f"- **[{r['priority']}] {r['type']}**, {r.get('advice', '')}")

        # ----- Per-requirement semantic table -----
        with st.expander("Per JD requirement, semantic match detail"):
            st.caption(
                "For every JD sentence, the most similar resume sentence and "
                "their cosine similarity. Rows with low similarity are JD "
                "requirements that no resume bullet currently addresses."
            )
            for row in result.semantic["per_requirement"][:30]:
                st.markdown(f"- **sim={row['similarity']:.2f}**, JD: {row['jd_sentence'][:140]}")
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
    st.caption(
        "Tracks your Resume Score across edits. Sub-scores are shown on a "
        "0–100 scale (sub-score × 100) so they share an axis with the final "
        "score."
    )
    import pandas as pd
    df = pd.DataFrame([
        {
            "version": h["label"],
            "Resume Score": h["final_score"],
            "keyword (×100)": h["sub_scores"]["keyword"] * 100,
            "semantic (×100)": h["sub_scores"]["semantic"] * 100,
            "quality (×100)": h["sub_scores"]["quality"] * 100,
        }
        for h in st.session_state.history
    ])
    st.line_chart(df.set_index("version"))
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------- Footer: what the Resume Score is and is not ----------
with st.expander("About the Resume Score, what it measures and what it doesn't"):
    st.markdown(
        """
**What the Resume Score is.** A transparent 0–100 number computed from three
deterministic components: TF-IDF keyword overlap, Sentence-BERT semantic
similarity between JD requirements and resume bullets, and a rule-based
content audit. The breakdown table above shows the exact arithmetic, the
explanations in ResumeMax are generated directly from the scoring
components and therefore remain fully traceable to the system's
computations.

**What it isn't.** The Resume Score is **not** a prediction of
employability, interview probability, or hiring outcomes. Adding a missing
skill such as Kubernetes might raise your Resume Score by, say, 4–5
points, but that is a change in *this system's keyword/semantic alignment
metric*, not a change in your real-world chances.

#### How the Resume Score is validated

ResumeMax is evaluated on four levels so that any reported correlation with
human ratings is interpretable rather than a single isolated number.

1. **Pilot gold set (n = 8 pairs, 4 role families).** Two synthetic resumes
   per family (intentionally weak vs. optimized) are annotated by a senior
   data-science practitioner on a 0–100 rubric covering overall fit, skill
   coverage, and writing polish. The set is shipped with the repo so any
   reviewer can reproduce the numbers. Current pilot result on the headline
   axis: Pearson r ≈ 0.96, Spearman ρ ≈ 0.88, MAE ≈ 15.

2. **Inter-annotator agreement (Krippendorff's α, interval level).** The
   evaluation pipeline supports multi-annotator CSV columns
   (`human_fit_A1`, `human_fit_A2`, …) and reports α per rubric axis. The
   shipped pilot has a single annotator and the tool reports that status
   explicitly rather than emitting a misleading number; the planned
   extension is a 50-pair set scored by two senior practitioners,
   targeting α ≥ 0.70.

3. **Per-component ablation (TF-IDF only / SBERT only / quality only /
   full system).** Every gold pair is re-scored under four weight
   schedules so the marginal contribution of each component is visible
   in any results table. On the pilot the full system matches or beats
   every single-component baseline on Pearson r against human ratings,
   which is the empirical justification for the default weight schedule
   (keyword 0.35 / semantic 0.45 / quality 0.20) rather than a hand-wave.

4. **Large-scale validation (~6 000 pairs).** The same four-schedule
   ablation is re-run on a public resume / job-description corpus
   (`cnamuangtoun/resume-job-description-fit` on HuggingFace by default,
   with Kaggle and bring-your-own-CSV modes also supported). Both the
   human labels and ResumeMax scores are min-max rescaled to 0–100, and
   Pearson r is reported with a 1 000-iteration bootstrap 95 % confidence
   interval, so a small fluctuation between runs is not over-interpreted.

**Reporting rule.** Any quoted full-system Pearson r is always accompanied
by the TF-IDF-only and SBERT-only Pearson r on the same data, the sample
size, the annotator count, and (where ≥ 2 annotators exist) Krippendorff's
α. Score deltas are always phrased as "Resume Score points", never as
employability or interview-rate predictions.
        """
    )
