<div align="center">

# 🧠 ResumeFit AI
### AI-Powered Resume Job Match & Skill Gap Predictor

**Instantly see how well a resume matches a job — with a fit score, predicted job category, and a full skill gap breakdown.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://salma-resumefit-ai.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-8A2BE2?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-2E8B57?style=for-the-badge)

**🔗 Try it live: [salma-resumefit-ai.streamlit.app](https://salma-resumefit-ai.streamlit.app/)**

</div>

---

## 📌 Overview

**ResumeFit AI** is an end-to-end NLP + Machine Learning project that helps candidates understand how well their resume aligns with a specific job description — the same way an Applicant Tracking System (ATS) would screen it — while keeping every calculation transparent and explainable.

It combines three components into one Streamlit dashboard:

| Component | What it does | Technique |
|---|---|---|
| 🏷️ **Job Category Prediction** | Classifies a resume into a job category (e.g. Data Scientist, HR, Sales) | TF-IDF + Logistic Regression, trained on the Kaggle Resume Dataset |
| 🎯 **Resume–JD Match Score** | Scores how well a resume matches a specific job description | TF-IDF + Cosine Similarity, blended with skill-keyword coverage |
| 🧩 **Skill Gap Analysis** | Shows exactly which required skills are present vs. missing | Keyword-based extraction across a 280+ skill dictionary spanning tech, business, HR, finance, healthcare, and more |

---

## ✨ Features

- 📄 Upload a resume as a **PDF** — text is extracted automatically
- 📝 Paste any **job description** to compare against
- 📊 A single **Match Score** blending keyword overlap and text similarity, scaled to feel like real ATS tools
- 🏷️ **Predicted job category** with confidence, powered by a trained classifier
- ✅ **Matching skills** and ⚠️ **missing skills**, shown as clear visual chips
- 📈 Progress bars for skill coverage and text similarity
- 🎨 A clean, card-based dashboard — no clutter, no tutorial-style walls of text

---

## 🖥️ Live Demo

**👉 [salma-resumefit-ai.streamlit.app](https://salma-resumefit-ai.streamlit.app/)**

Upload a resume PDF, paste a job description, and hit **Analyze** to see it in action.

---

## 🗂️ Project Structure

```
Resume_Job_Match_Predictor/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt            # Python dependencies
├── README.md                   # You are here
├── INTERVIEW_PREP.md           # Full technical explanation + interview Q&A
├── .gitignore
├── data/
│   └── README.md               # Instructions to obtain Resume.csv (not included)
├── models/
│   └── README.md                # Trained artifacts land here after training
├── notebooks/
│   └── model_training.ipynb    # Full training + evaluation walkthrough
└── src/
    ├── __init__.py
    ├── preprocessing.py         # Text cleaning utilities
    ├── pdf_extractor.py         # PDF resume text extraction (pypdf)
    ├── skill_extractor.py       # 280+ skill dictionary + extraction logic
    ├── matcher.py                # TF-IDF + cosine similarity + score curve
    └── predictor.py              # Training pipeline + inference (job category)
```

---

## ⚙️ Setup

### 1️⃣ Clone & create a virtual environment

```bash
git clone <your-repo-url>
cd Resume_Job_Match_Predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset

This repo does **not** include `Resume.csv`. Download it from Kaggle:

🔗 https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Place it at:

```
data/Resume.csv
```

Required columns: `ID`, `Resume_str`, `Resume_html`, `Category`.

---

## 🏋️ Train the Model

Trained model artifacts are **not committed to git** (see `.gitignore`) — you train once, locally.

### Option A — Command line (fastest)

```bash
python -m src.predictor --train
```

This loads and cleans `data/Resume.csv`, splits into train/test, fits a TF-IDF vectorizer, trains a Logistic Regression classifier, prints accuracy/precision/recall/F1, and saves everything to `models/`.

### Option B — Jupyter Notebook (with visuals)

```bash
jupyter notebook notebooks/model_training.ipynb
```

Includes dataset exploration, a cleaning preview, the classification report as a table, and a confusion matrix heatmap.

> ⚠️ **If deploying** (e.g. to Streamlit Community Cloud), the platform never runs training for you — you must train locally and **force-add** the model files past `.gitignore`:
> ```bash
> git add -f models/job_category_model.pkl models/tfidf_vectorizer.pkl models/label_encoder.pkl models/metrics.json
> git commit -m "Add trained model artifacts for deployment"
> git push
> ```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`), or use the [live demo](https://salma-resumefit-ai.streamlit.app/).

---

## 🔬 ML / NLP Methodology

### 🏷️ Job Category Prediction
- **Input:** `Resume_str` column from `Resume.csv`
- **Cleaning:** lowercasing, URL/email removal, punctuation & digit stripping, whitespace normalization, stopword removal (`src/preprocessing.py`)
- **Vectorization:** `TfidfVectorizer` (unigrams + bigrams, 5,000-feature vocabulary), fit on the training split only — avoids data leakage
- **Model:** `LogisticRegression` (multinomial, `max_iter=1000`)
- **Evaluation:** accuracy, weighted precision/recall/F1, full per-class classification report, and a confusion matrix — all on a stratified 80/20 held-out test split
- **Persistence:** vectorizer, model, and label encoder saved with `joblib`

### 🎯 Resume–JD Matching
Two signals are combined:
1. **Text similarity** — a fresh TF-IDF vectorizer fit on just the resume + JD pair, compared with cosine similarity
2. **Skill coverage** — % of the JD's detected required skills that also appear in the resume

```
raw_score  = 0.4 × text_similarity + 0.6 × skill_coverage
final_score = 100 × (raw_score / 100) ^ 0.6      # display curve
```

The curve mirrors how commercial ATS-style tools present scores — raw lexical overlap between two documents is mathematically almost never near 100%, so the result is rescaled into a friendlier, still fully transparent range.

### 🧩 Skill Gap Analysis
- A curated **280+ skill dictionary** (`src/skill_extractor.py`) spanning programming, data science, cloud/DevOps, business, finance, marketing, HR, legal, healthcare, education, hospitality, manufacturing, admin, and soft skills
- Word-boundary-safe, case-insensitive regex matching — correctly handles multi-word skills ("machine learning") and symbol-containing tools ("c++", "node.js")
- Reports **matching skills**, **missing skills**, and a **skill match percentage**

---

## ⚠️ Limitations

- The Match Score is a **text-similarity metric**, not a hiring probability — it doesn't capture experience depth, seniority, soft skills, or interview performance
- Skill extraction is **keyword-based** — unusual phrasing or unlisted synonyms won't be detected
- Category prediction quality depends on how balanced `Resume.csv`'s categories are
- Scanned/image-only PDFs with no text layer won't extract any text

---

## 🚀 Future Enhancements

- **Semantic similarity** via Sentence Transformers (e.g. `all-MiniLM-L6-v2`) instead of pure lexical TF-IDF overlap
- Expand the skills dictionary with a maintained taxonomy (ESCO / O*NET) + fuzzy/synonym matching
- Resume section parsing (Experience, Education, Skills) for more targeted analysis
- DOCX resume upload support
- Recruiter feedback loop to continuously retrain the classifier

---

## 📋 Quick Command Reference

| Task | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Train the model | `python -m src.predictor --train` |
| Train via notebook | `jupyter notebook notebooks/model_training.ipynb` |
| Run the app | `streamlit run app.py` |

---

## 📚 Learn More

See **[INTERVIEW_PREP.md](INTERVIEW_PREP.md)** for a full technical walkthrough of this project, including how every module works and commonly asked interview questions with model answers.

---

<div align="center">

Built with 🐍 Python · 🔬 scikit-learn · 🎈 Streamlit

</div>
