"""
app.py

AI-Powered Resume Job Match & Skill Gap Predictor
--------------------------------------------------
A Streamlit dashboard that:
    1. Predicts the most likely job category for an uploaded resume
       (Logistic Regression trained on Resume.csv via TF-IDF features).
    2. Computes a Resume <-> Job Description match score using
       TF-IDF + cosine similarity.
    3. Performs a skill-gap analysis (matching vs. missing skills).

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so `src` imports work regardless
# of the working directory Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matcher import compute_text_similarity, match_summary
from src.pdf_extractor import extract_text_from_pdf
from src.predictor import artifacts_exist, load_metrics, predict_category
from src.skill_extractor import compare_skills

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeFit AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }

    .hero {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.2rem;
    }
    .hero-icon {
        font-size: 2.3rem;
        line-height: 1;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #111827;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #6b7280;
        margin: 4px 0 1.6rem 0;
    }

    .card {
        background: #ffffff;
        border: 1px solid #eceef1;
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        margin-bottom: 1.2rem;
    }

    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    div[data-testid="stMetric"] {
        background: #f9fafb;
        border: 1px solid #f0f1f3;
        border-radius: 12px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 500;
    }

    .skill-pill {
        display: inline-block;
        padding: 5px 13px;
        margin: 3px 4px 3px 0;
        border-radius: 999px;
        font-size: 0.83rem;
        font-weight: 500;
    }
    .pill-match {
        background-color: #ecfdf3;
        color: #166534;
        border: 1px solid #b7ebc6;
    }
    .pill-missing {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    .muted-caption {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 1.4rem;
        text-align: center;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🧠</div>
        <div class="hero-title">ResumeFit AI</div>
    </div>
    <div class="hero-subtitle">
        Instantly see how well a resume matches a job — with a fit score, predicted role, and a skill gap breakdown.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 ResumeFit AI")
    st.caption("Resume-to-job matching, powered by NLP.")
    st.markdown("---")

    metrics = load_metrics()
    if metrics:
        st.markdown("**Model performance**")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.0f}%")
        m2.metric("F1 score", f"{metrics['f1_weighted'] * 100:.0f}%")
        st.caption(f"{metrics['n_classes']} job categories · {metrics['n_train_samples']} training resumes")
        st.markdown("---")

    with st.expander("How scoring works"):
        st.caption(
            "The Match Score blends two signals: how many of the job's "
            "required skills appear in the resume (weighted highest), and "
            "overall TF-IDF text similarity between the two documents. "
            "It reflects content and keyword overlap, not a guarantee of "
            "being hired — experience level, soft skills, and interview "
            "performance aren't captured."
        )

# --------------------------------------------------------------------------
# Input section
# --------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📄 Resume & Job Description</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    uploaded_pdf = st.file_uploader("Resume (PDF)", type=["pdf"], label_visibility="visible")
    resume_text = ""
    if uploaded_pdf is not None:
        try:
            resume_text = extract_text_from_pdf(uploaded_pdf)
            if resume_text:
                st.caption(f"✓ {len(resume_text.split())} words extracted")
                with st.expander("Preview extracted text"):
                    st.text_area("Resume text", resume_text, height=180, label_visibility="collapsed")
            else:
                st.error("No text could be extracted — this PDF may be a scanned image.")
        except ValueError as exc:
            st.error(str(exc))

with col_right:
    jd_text = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste the job description here...",
        label_visibility="visible",
    )

analyze_clicked = st.button("Analyze Match", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Analysis section
# --------------------------------------------------------------------------
if analyze_clicked:
    if not resume_text.strip():
        st.error("Please upload a resume PDF with extractable text before analyzing.")
    elif not jd_text.strip():
        st.error("Please paste a job description before analyzing.")
    else:
        score, label = match_summary(resume_text, jd_text)
        text_score = compute_text_similarity(resume_text, jd_text)
        prediction = predict_category(resume_text)
        skills_result = compare_skills(resume_text, jd_text)

        # ---- Overview metrics ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Overview</div>', unsafe_allow_html=True)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Match Score", f"{score:.0f}%", label)
        metric_cols[1].metric("Predicted Role", prediction["predicted_category"] if prediction else "—")
        metric_cols[2].metric("Skill Coverage", f"{skills_result['match_percentage']:.0f}%")
        metric_cols[3].metric("Skills Detected", len(skills_result["resume_skills"]))

        st.write("")
        st.progress(min(int(skills_result["match_percentage"]), 100), text=f"Required skills covered — {skills_result['match_percentage']:.0f}%")
        st.progress(min(int(text_score), 100), text=f"Overall text similarity — {text_score:.0f}%")

        if prediction and len(prediction["top_predictions"]) > 1:
            with st.expander("Other likely role matches"):
                top_df = pd.DataFrame(prediction["top_predictions"][1:], columns=["Category", "Probability"])
                top_df["Probability"] = (top_df["Probability"] * 100).round(0).astype(int).astype(str) + "%"
                st.dataframe(top_df, hide_index=True, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Skills ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Skill Gap Analysis</div>', unsafe_allow_html=True)

        skill_cols = st.columns(2)
        with skill_cols[0]:
            st.markdown("**Matching skills**")
            if skills_result["matching_skills"]:
                pills = "".join(
                    f'<span class="skill-pill pill-match">{s}</span>'
                    for s in skills_result["matching_skills"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("No overlapping skills found.")

        with skill_cols[1]:
            st.markdown("**Missing skills**")
            if skills_result["missing_skills"]:
                pills = "".join(
                    f'<span class="skill-pill pill-missing">{s}</span>'
                    for s in skills_result["missing_skills"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("Resume covers every skill found in the job description.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Summary ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Summary</div>', unsafe_allow_html=True)

        summary_cols = st.columns(3)
        summary_cols[0].metric("Resume length", f"{len(resume_text.split())} words")
        summary_cols[1].metric("Job description length", f"{len(jd_text.split())} words")
        summary_cols[2].metric(
            "Skills identified",
            len(set(skills_result["resume_skills"]) | set(skills_result["jd_skills"])),
        )

        with st.expander("Full skill breakdown"):
            max_len = max(len(skills_result["resume_skills"]), len(skills_result["jd_skills"]))
            resume_col = skills_result["resume_skills"] + [""] * (max_len - len(skills_result["resume_skills"]))
            jd_col = skills_result["jd_skills"] + [""] * (max_len - len(skills_result["jd_skills"]))
            breakdown_df = pd.DataFrame({"Resume Skills": resume_col, "Job Description Skills": jd_col})
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="muted-caption">Match score reflects text similarity, not hiring probability.</div>',
    unsafe_allow_html=True,
)
