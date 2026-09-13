"""
matcher.py

Resume <-> Job Description matching.

Two signals are combined into the final "Resume Match Score":

1. Text similarity  - TF-IDF vectors of the full resume/JD text compared
                       with cosine similarity. This captures overall
                       vocabulary/content overlap but is naturally low
                       (often 10-30%) even for good matches, since resumes
                       contain many words (dates, company names, formatting
                       artifacts) that never appear in a job description.

2. Skill coverage   - the fraction of skills required by the job
                       description that are actually present in the resume
                       (from src.skill_extractor). This is closer to how
                       most commercial "ATS score" tools actually work,
                       since they focus on keyword/skill matching rather
                       than whole-document similarity.

The raw blend of these two signals is then passed through a display curve
(see `apply_score_curve`) before being shown to the user. Most commercial
ATS-style checkers apply a similar curve internally — raw lexical/keyword
overlap between two documents is mathematically almost never close to
100%, so tools that show "78% match" are not measuring pure overlap, they
are rescaling it into a friendlier range. We do the same thing here, but
document it openly rather than hiding it.

This is still a similarity score, not a probability of getting hired.
"""

from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.skill_extractor import compare_skills

# Weight given to skill-keyword coverage vs. raw text similarity when
# computing the final blended match score.
SKILL_WEIGHT = 0.6
TEXT_WEIGHT = 0.4

# Exponent used to curve the final displayed score upward (0 < x < 1).
# A raw blended score is passed through score^CURVE_EXPONENT (after
# normalizing to 0-1), which lifts low/mid scores more than high scores
# while still mapping 0 -> 0 and 100 -> 100. Lower exponent = more generous.
CURVE_EXPONENT = 0.6


def apply_score_curve(raw_score: float, exponent: float = CURVE_EXPONENT) -> float:
    """
    Rescale a raw 0-100 similarity score into a more generous display score.

    Uses a power curve: displayed = 100 * (raw / 100) ** exponent

    This preserves 0% -> 0% and 100% -> 100%, but lifts everything in
    between (e.g. a raw 45% becomes ~62%, a raw 69% becomes ~80%), which
    mirrors how most commercial resume-matching tools present scores.

    Args:
        raw_score: The raw blended score (0-100).
        exponent: Curve strength; lower = more generous. Must be in (0, 1].

    Returns:
        Curved score (0-100), rounded to 2 decimals.
    """
    if raw_score <= 0:
        return 0.0
    if raw_score >= 100:
        return 100.0

    normalized = raw_score / 100.0
    curved = normalized ** exponent
    return round(curved * 100, 2)


def compute_text_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute raw TF-IDF + cosine similarity between resume and JD text.

    A fresh TF-IDF vectorizer is fit on just these two documents so the
    score reflects how similar the resume's vocabulary/content is to the
    specific job description provided.

    Args:
        resume_text: Raw text extracted from the candidate's resume.
        jd_text: Raw text of the job description.

    Returns:
        Similarity score as a percentage (0.0 - 100.0), rounded to 2 decimals.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    if not cleaned_resume or not cleaned_jd:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute the final (curved) Resume Match Score shown to the user.

    Steps:
        1. Blend text similarity and skill coverage:
           raw = TEXT_WEIGHT * text_similarity + SKILL_WEIGHT * skill_coverage
           (falls back to pure text similarity if the JD has no
           recognizable skill keywords at all)
        2. Apply a display curve (see `apply_score_curve`) so the result
           reads in a range comparable to typical ATS-style tools.

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.

    Returns:
        Match score as a percentage (0.0 - 100.0), rounded to 2 decimals.
    """
    text_score = compute_text_similarity(resume_text, jd_text)
    skills = compare_skills(resume_text, jd_text)

    if skills["jd_skills"]:
        raw = (TEXT_WEIGHT * text_score) + (SKILL_WEIGHT * skills["match_percentage"])
    else:
        raw = text_score

    return apply_score_curve(raw)


def match_summary(resume_text: str, jd_text: str) -> Tuple[float, str]:
    """
    Compute the match score and a short human-readable interpretation.

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.

    Returns:
        Tuple of (score, interpretation_label)
    """
    score = compute_match_score(resume_text, jd_text)

    if score >= 85:
        label = "Strong Match"
    elif score >= 70:
        label = "Good Match"
    elif score >= 50:
        label = "Moderate Match"
    else:
        label = "Weak Match"

    return score, label
