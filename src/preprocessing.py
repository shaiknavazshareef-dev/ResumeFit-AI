"""
preprocessing.py

Text cleaning and preprocessing utilities used both at training time
(on Resume_str from Resume.csv) and at inference time (on resume text
extracted from an uploaded PDF, and on pasted job descriptions).

No external NLP corpora (e.g. NLTK downloads) are required, so the
project works fully offline out of the box.
"""

import re

# A compact, practical English stopword list. Kept local (no downloads)
# so the app works without internet access at runtime.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves", "im", "ive", "id", "youre", "youve", "dont", "didnt",
}

_URL_PATTERN = re.compile(r"http\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean raw resume / job description text for TF-IDF vectorization.

    Steps:
        1. Lowercase the text
        2. Remove URLs and email addresses
        3. Remove digits, punctuation, and special characters
        4. Collapse extra whitespace
        5. Optionally remove common English stopwords

    Args:
        text: Raw input text.
        remove_stopwords: Whether to strip stopwords.

    Returns:
        Cleaned, lowercase text string ready for vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()

    if remove_stopwords:
        tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
        text = " ".join(tokens)

    return text


def clean_series(series):
    """
    Apply clean_text to a pandas Series of raw resume text.

    Args:
        series: pandas Series of strings.

    Returns:
        pandas Series of cleaned strings.
    """
    return series.astype(str).apply(clean_text)
