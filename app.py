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

from src.matcher import match_summary
from src.pdf_extractor import extract_text_from_pdf
from src.predictor import artifacts_exist, load_metrics, predict_category
from src.skill_extractor import compare_skills

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Job Match & Skill Gap Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.4rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.3rem;
    }
    .skill-pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .pill-match {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .pill-missing {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .footer-note {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 2rem;
        border-top: 1px solid #e5e7eb;
        padding-top: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 AI-Powered Resume Job Match & Skill Gap Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">TF-IDF + Logistic Regression job-category prediction, '
    'TF-IDF/cosine-similarity resume-to-JD matching, and keyword-based skill gap analysis.</div>',
    unsafe_allow_html=True,
)

if not artifacts_exist():
    st.warning(
        "⚠️ No trained model found in `models/`. The **Predicted Job Category** section "
        "will be unavailable until you train the model. See the README for training "
        "instructions (`python -m src.predictor --train`)."
    )

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 About")
    st.write(
        "This tool analyzes a candidate's resume against a target job "
        "description and reports:\n"
        "- Predicted job category\n"
        "- Resume-to-JD match score\n"
        "- Matching & missing skills"
    )
    st.markdown("---")
    metrics = load_metrics()
    if metrics:
        st.subheader("📈 Model Performance")
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        st.metric("F1 (weighted)", f"{metrics['f1_weighted'] * 100:.1f}%")
        st.caption(
            f"Trained on {metrics['n_train_samples']} samples · "
            f"Tested on {metrics['n_test_samples']} samples · "
            f"{metrics['n_classes']} categories"
        )
    else:
        st.caption("Train the model to see performance metrics here.")
    st.markdown("---")
    st.caption(
        "⚠️ The match score is a **text-similarity score**, not a probability "
        "of getting hired. See README for details."
    )

# --------------------------------------------------------------------------
# Input section
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">1️⃣ Upload Resume & Job Description</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    resume_text = ""
    if uploaded_pdf is not None:
        try:
            resume_text = extract_text_from_pdf(uploaded_pdf)
            if resume_text:
                st.success(f"✅ Extracted {len(resume_text.split())} words from resume.")
                with st.expander("Preview extracted resume text"):
                    st.text_area("Resume text", resume_text, height=200, label_visibility="collapsed")
            else:
                st.error("No extractable text found in this PDF (it may be a scanned image).")
        except ValueError as exc:
            st.error(str(exc))

with col_right:
    jd_text = st.text_area(
        "Paste the Job Description",
        height=260,
        placeholder="Paste the full job description here...",
    )

analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Analysis section
# --------------------------------------------------------------------------
if analyze_clicked:
    if not resume_text.strip():
        st.error("Please upload a resume PDF with extractable text before analyzing.")
    elif not jd_text.strip():
        st.error("Please paste a job description before analyzing.")
    else:
        st.markdown('<div class="section-title">2️⃣ Results</div>', unsafe_allow_html=True)

        # ---- Match Score ----
        score, label = match_summary(resume_text, jd_text)

        # ---- Job Category Prediction ----
        prediction = predict_category(resume_text)

        # ---- Skill Gap Analysis ----
        skills_result = compare_skills(resume_text, jd_text)

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Resume Match Score", f"{score:.1f}%", label)
        with metric_cols[1]:
            if prediction:
                st.metric("Predicted Job Category", prediction["predicted_category"])
            else:
                st.metric("Predicted Job Category", "N/A")
        with metric_cols[2]:
            st.metric("Skill Match", f"{skills_result['match_percentage']:.1f}%")
        with metric_cols[3]:
            st.metric("Skills Found (Resume)", len(skills_result["resume_skills"]))

        st.progress(min(int(score), 100), text=f"Overall Resume–JD Similarity: {score:.1f}%")
        st.progress(min(int(skills_result["match_percentage"]), 100), text=f"Skill Coverage: {skills_result['match_percentage']:.1f}%")

        if prediction:
            st.markdown("**Top predicted categories:**")
            top_df = pd.DataFrame(prediction["top_predictions"], columns=["Category", "Probability"])
            top_df["Probability"] = (top_df["Probability"] * 100).round(1).astype(str) + "%"
            st.table(top_df)

        st.markdown("---")

        skill_cols = st.columns(2)
        with skill_cols[0]:
            st.markdown("#### ✅ Matching Skills")
            if skills_result["matching_skills"]:
                pills = "".join(
                    f'<span class="skill-pill pill-match">{s}</span>'
                    for s in skills_result["matching_skills"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.info("No overlapping skills detected between resume and job description.")

        with skill_cols[1]:
            st.markdown("#### ⚠️ Missing Skills")
            if skills_result["missing_skills"]:
                pills = "".join(
                    f'<span class="skill-pill pill-missing">{s}</span>'
                    for s in skills_result["missing_skills"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.success("No missing skills detected — resume covers all identified JD skills!")

        st.markdown("---")
        st.markdown('<div class="section-title">3️⃣ Resume Analysis Summary</div>', unsafe_allow_html=True)

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.write("**Resume word count**")
            st.write(len(resume_text.split()))
        with summary_cols[1]:
            st.write("**Job description word count**")
            st.write(len(jd_text.split()))
        with summary_cols[2]:
            st.write("**Total distinct skills identified**")
            st.write(len(set(skills_result["resume_skills"]) | set(skills_result["jd_skills"])))

        with st.expander("Full skill breakdown (resume vs. job description)"):
            max_len = max(len(skills_result["resume_skills"]), len(skills_result["jd_skills"]))
            resume_col = skills_result["resume_skills"] + [""] * (max_len - len(skills_result["resume_skills"]))
            jd_col = skills_result["jd_skills"] + [""] * (max_len - len(skills_result["jd_skills"]))
            breakdown_df = pd.DataFrame({"Resume Skills": resume_col, "Job Description Skills": jd_col})
            st.dataframe(breakdown_df, use_container_width=True)

st.markdown(
    '<div class="footer-note">⚠️ <b>Disclaimer:</b> The match score is a text-similarity metric '
    'derived from TF-IDF and cosine similarity. It reflects vocabulary/content overlap between '
    'the resume and the job description — it is <b>not</b> a probability of getting hired, and '
    'does not account for experience level, soft skills, interview performance, or fit beyond '
    'textual similarity.</div>',
    unsafe_allow_html=True,
)
