# AI-Powered Resume Job Match & Skill Gap Predictor

An end-to-end NLP + Machine Learning project that:

1. **Predicts a resume's job category** using TF-IDF + Logistic Regression, trained on the Kaggle Resume Dataset.
2. **Scores how well a resume matches a job description** using TF-IDF + cosine similarity.
3. **Performs a skill gap analysis**, showing matching and missing technical skills.
4. Presents everything in a polished, interactive **Streamlit dashboard**.

---

## Project Structure

```
Resume_Job_Match_Predictor/
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md               # Instructions to obtain Resume.csv (not included)
├── models/
│   └── README.md               # Trained artifacts saved here after training
├── notebooks/
│   └── model_training.ipynb    # Full training + evaluation walkthrough
└── src/
    ├── __init__.py
    ├── preprocessing.py         # Text cleaning utilities
    ├── pdf_extractor.py         # PDF resume text extraction (pypdf)
    ├── skill_extractor.py       # Skills dictionary + extraction logic
    ├── matcher.py                # TF-IDF + cosine similarity matching
    └── predictor.py              # Training pipeline + inference (job category)
```

---

## 1. Setup

### 1.1 Clone the repository and create a virtual environment

```bash
git clone <your-repo-url>
cd Resume_Job_Match_Predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Download the dataset

This repository does **not** include `Resume.csv`. Download it from Kaggle:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Place the file at:

```
data/Resume.csv
```

Required columns: `ID`, `Resume_str`, `Resume_html`, `Category`.

---

## 2. Train the Model

You must train the model once before running the app (trained artifacts are not
committed to the repo — see `models/README.md`).

### Option A — Command line (recommended, fastest)

```bash
python -m src.predictor --train
```

This will:
- Load and clean `data/Resume.csv`
- Split into train/test sets
- Fit a TF-IDF vectorizer
- Train a Logistic Regression classifier
- Print accuracy, precision, recall, and F1-score
- Save `job_category_model.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`, and `metrics.json` into `models/`

### Option B — Jupyter Notebook (with visualizations)

```bash
jupyter notebook notebooks/model_training.ipynb
```

Run all cells to see dataset exploration, cleaning previews, the classification
report as a table, and a confusion matrix heatmap, in addition to saving the
same artifacts to `models/`.

---

## 3. Run the Streamlit App

Once the model is trained (artifacts exist in `models/`):

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

### Using the app
1. Upload a resume as a **PDF**.
2. Paste the target **job description** into the text area.
3. Click **Analyze**.
4. Review the Resume Match Score, Predicted Job Category, Matching/Missing
   Skills, Skill Match percentage, and the resume analysis summary.

---

## 4. ML / NLP Methodology

### 4.1 Job Category Prediction
- **Input:** `Resume_str` column from `Resume.csv`.
- **Cleaning:** lowercasing, URL/email removal, punctuation & digit removal,
  whitespace normalization, and stopword removal (`src/preprocessing.py`).
- **Vectorization:** `TfidfVectorizer` (unigrams + bigrams, capped vocabulary
  of 5,000 features) fit on the training split only, to avoid data leakage.
- **Model:** `LogisticRegression` (multinomial, `max_iter=1000`), trained on
  the TF-IDF features.
- **Evaluation:** accuracy, weighted precision, weighted recall, weighted
  F1-score, full per-class classification report, and a confusion matrix —
  all computed on a held-out test split (default 80/20, stratified by class).
- **Persistence:** the fitted vectorizer, model, and label encoder are saved
  with `joblib` so inference in the Streamlit app uses the exact training-time
  vocabulary and class mapping.

### 4.2 Resume–Job Description Matching
- Text is extracted from the uploaded PDF using `pypdf`.
- Both the resume text and job description are cleaned with the same
  preprocessing pipeline.
- A **fresh** `TfidfVectorizer` is fit at query time on just these two
  documents, and **cosine similarity** between their TF-IDF vectors produces
  the Resume Match Score (0–100%).

### 4.3 Skill Gap Analysis
- A curated technical-skills dictionary (`src/skill_extractor.py`), organized
  by domain (programming languages, web dev, data science/ML, data
  engineering, databases, cloud/DevOps, BI tools, project/soft skills).
- Skills are extracted from both texts using word-boundary-safe, case-insensitive
  regex matching (supports multi-word skills like "machine learning" and
  symbol-containing skills like "c++" / "node.js").
- The app reports **matching skills**, **missing skills** (present in the JD
  but absent from the resume), and a **skill match percentage**
  (`matching / required-by-JD`).

---

## 5. Limitations

- **The Resume Match Score is a text-similarity score, not a hiring
  probability.** It reflects vocabulary and content overlap between the
  resume and job description as measured by TF-IDF + cosine similarity. It
  does **not** account for years of experience, seniority, soft skills,
  interview performance, cultural fit, or anything not captured in the
  literal wording of both documents.
- Skill extraction is **keyword-based**. It will miss skills phrased in
  unusual ways, synonyms not in the dictionary, or skills implied but not
  explicitly named.
- Job category prediction quality depends entirely on the diversity and
  balance of categories in the training dataset (`Resume.csv`). Categories
  with very few training examples will predict less reliably.
- PDF extraction quality depends on how the PDF was generated; scanned
  image-based PDFs with no embedded text layer will not extract any text.

---

## 6. Future Enhancements

- Replace/augment TF-IDF matching with **Sentence Transformers** (e.g.
  `all-MiniLM-L6-v2` from `sentence-transformers`) to capture **semantic
  similarity** rather than pure lexical overlap, improving matches when a
  resume and JD describe the same skill with different wording.
- Expand the skills dictionary using a maintained taxonomy (e.g. ESCO or
  O*NET) and support fuzzy/synonym matching.
- Add resume section parsing (Experience, Education, Skills) for more
  targeted analysis instead of treating the resume as one block of text.
- Support DOCX resume uploads in addition to PDF.
- Add a feedback loop where recruiters can correct predicted categories to
  continuously improve the classifier.

---

## 7. Quick Command Reference

| Task                          | Command                                  |
|--------------------------------|-------------------------------------------|
| Install dependencies           | `pip install -r requirements.txt`         |
| Train the model                | `python -m src.predictor --train`         |
| Train via notebook             | `jupyter notebook notebooks/model_training.ipynb` |
| Run the app                    | `streamlit run app.py`                    |
