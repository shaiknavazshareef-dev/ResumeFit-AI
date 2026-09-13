"""
predictor.py

Loads the trained TF-IDF vectorizer, Logistic Regression classifier, and
label encoder (produced by training on Resume.csv) and exposes a simple
`predict_category` function for inference inside the Streamlit app.

Also exposes `train_and_save_model`, the single source of truth for the
training pipeline, so both the notebook and any script can call the exact
same real training logic (no duplicated / fake logic).
"""

import json
import os
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import clean_series

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "job_category_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "Resume.csv")


def train_and_save_model(
    data_path: str = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
) -> Dict:
    """
    Train the job-category classifier from Resume.csv and persist artifacts.

    Pipeline:
        1. Load Resume.csv (columns: ID, Resume_str, Resume_html, Category)
        2. Clean Resume_str text
        3. Encode Category labels
        4. Train/test split
        5. Fit TF-IDF vectorizer on the training text
        6. Train a Logistic Regression classifier
        7. Evaluate on the held-out test set (accuracy, precision, recall, F1,
           confusion matrix, full classification report)
        8. Save the model, vectorizer, label encoder, and metrics to disk

    Args:
        data_path: Path to Resume.csv.
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.
        max_features: Max vocabulary size for TF-IDF.

    Returns:
        Dictionary of evaluation metrics (also saved to models/metrics.json).

    Raises:
        FileNotFoundError: If Resume.csv is not found at data_path.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Download Resume.csv from "
            "Kaggle and place it in the data/ folder (see data/README.md)."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(data_path)
    required_cols = {"ID", "Resume_str", "Resume_html", "Category"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Resume.csv is missing required columns: {missing_cols}")

    df = df.dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)

    # 1. Clean text
    df["cleaned_resume"] = clean_series(df["Resume_str"])
    df = df[df["cleaned_resume"].str.strip() != ""].reset_index(drop=True)

    # 2. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    # 3. Train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["cleaned_resume"],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 4. TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 5. Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": conf_matrix,
        "labels": label_encoder.classes_.tolist(),
        "classification_report": report,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_classes": int(len(label_encoder.classes_)),
    }

    # 7. Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def artifacts_exist() -> bool:
    """Check whether trained model artifacts are present on disk."""
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(VECTORIZER_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    )


def load_artifacts():
    """
    Load the trained model, TF-IDF vectorizer, and label encoder from disk.

    Returns:
        Tuple of (model, vectorizer, label_encoder)

    Raises:
        FileNotFoundError: If artifacts have not been trained/saved yet.
    """
    if not artifacts_exist():
        raise FileNotFoundError(
            "Trained model artifacts not found in models/. "
            "Run the training notebook or `python -m src.predictor --train` first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, label_encoder


def predict_category(text: str, top_k: int = 3) -> Optional[Dict]:
    """
    Predict the job category for a given resume text using the trained model.

    Args:
        text: Raw resume text (will be cleaned internally).
        top_k: Number of top candidate categories (with probabilities) to return.

    Returns:
        Dictionary with:
            - "predicted_category": most likely category (str)
            - "confidence": probability of the top prediction (float, 0-1)
            - "top_predictions": list of (category, probability) tuples
        Returns None if artifacts are not available or text is empty.
    """
    from src.preprocessing import clean_text  # local import avoids circulars at module load

    if not text or not text.strip():
        return None

    if not artifacts_exist():
        return None

    model, vectorizer, label_encoder = load_artifacts()

    cleaned = clean_text(text)
    if not cleaned:
        return None

    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]

    top_indices = probabilities.argsort()[::-1][:top_k]
    top_predictions = [
        (label_encoder.inverse_transform([idx])[0], round(float(probabilities[idx]), 4))
        for idx in top_indices
    ]

    predicted_category = top_predictions[0][0]
    confidence = top_predictions[0][1]

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


def load_metrics() -> Optional[Dict]:
    """Load previously saved training metrics, if available."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the job-category classifier.")
    parser.add_argument("--train", action="store_true", help="Train and save the model.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to Resume.csv")
    args = parser.parse_args()

    if args.train:
        results = train_and_save_model(data_path=args.data)
        print(json.dumps({k: v for k, v in results.items() if k != "classification_report"}, indent=2))
    else:
        parser.print_help()
