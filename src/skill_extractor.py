"""
skill_extractor.py

A practical, cross-industry skills dictionary and simple, dependency-free
keyword-based skill extraction used for the Skill Gap Analysis feature.

Matching is done with word-boundary-aware regex over the lowercased text,
so multi-word skills like "machine learning" or "power bi" are matched
correctly and short skills like "r" or "go" don't false-positive on
substrings of other words.

The dictionary intentionally covers both technical domains (programming,
data, cloud) and non-technical professional domains (business, finance,
marketing, HR, legal, healthcare, hospitality, etc.) since resume datasets
like the Kaggle Resume Dataset span many industries, not just tech.
"""

import re
from typing import Dict, List, Set

SKILLS_DB: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "javascript", "typescript", "r",
        "go", "golang", "ruby", "php", "swift", "kotlin", "scala", "rust",
        "matlab", "perl", "sql", "bash", "shell scripting", "vba",
    ],
    "Web Development": [
        "html", "css", "react", "angular", "vue", "node.js", "nodejs",
        "django", "flask", "fastapi", "spring boot", "asp.net", "next.js",
        "rest api", "graphql", "bootstrap", "tailwind css", "jquery",
    ],
    "Data Science & ML": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "data analysis", "data visualization",
        "statistics", "pandas", "numpy", "scikit-learn", "tensorflow",
        "keras", "pytorch", "xgboost", "opencv", "feature engineering",
        "predictive modeling", "time series analysis", "a/b testing",
    ],
    "Data Engineering & Big Data": [
        "spark", "hadoop", "kafka", "airflow", "etl", "data pipeline",
        "data warehousing", "hive", "databricks", "snowflake",
        "big data", "dbt",
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "oracle", "sqlite",
        "microsoft sql server", "redis", "cassandra", "dynamodb",
        "elasticsearch",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "google cloud platform", "gcp", "docker",
        "kubernetes", "terraform", "jenkins", "ci/cd", "ansible",
        "linux", "git", "github", "gitlab", "devops", "microservices",
    ],
    "BI & Visualization Tools": [
        "power bi", "tableau", "excel", "microsoft excel", "looker",
        "qlikview", "google analytics", "google data studio", "spreadsheets",
    ],
    "Office & Productivity": [
        "microsoft office", "powerpoint", "microsoft word", "outlook",
        "google workspace", "google sheets", "slack", "notion", "sharepoint",
    ],
    "Project & Program Management": [
        "agile", "scrum", "kanban", "jira", "project management",
        "program management", "risk management", "budgeting", "scheduling",
        "stakeholder management", "process improvement", "six sigma",
        "lean", "pmp",
    ],
    "Business & Management": [
        "strategic planning", "business development", "operations management",
        "vendor management", "supply chain management", "logistics",
        "procurement", "inventory management", "quality assurance",
        "quality control", "kpi tracking", "performance management",
        "change management", "business analysis", "market research",
        "forecasting", "cost reduction", "process optimization",
    ],
    "Finance & Accounting": [
        "financial analysis", "financial reporting", "budgeting",
        "accounts payable", "accounts receivable", "bookkeeping",
        "general ledger", "reconciliation", "auditing", "tax preparation",
        "payroll", "quickbooks", "sap", "cost accounting", "gaap",
        "corporate finance", "financial modeling", "invoicing",
    ],
    "Marketing & Sales": [
        "digital marketing", "content marketing", "social media marketing",
        "seo", "sem", "email marketing", "brand management", "copywriting",
        "market analysis", "lead generation", "crm", "salesforce",
        "negotiation", "client relationship management", "sales strategy",
        "cold calling", "account management", "customer acquisition",
        "campaign management", "public relations", "advertising",
    ],
    "Human Resources": [
        "recruitment", "talent acquisition", "onboarding", "employee relations",
        "performance appraisal", "compensation and benefits", "hr policy",
        "training and development", "workforce planning", "hris",
        "conflict resolution", "employee engagement", "labor relations",
        "succession planning",
    ],
    "Legal & Compliance": [
        "contract negotiation", "regulatory compliance", "legal research",
        "litigation", "corporate law", "contract drafting", "due diligence",
        "risk assessment", "compliance monitoring", "intellectual property",
    ],
    "Healthcare": [
        "patient care", "clinical documentation", "medical terminology",
        "electronic health records", "ehr", "emr", "hipaa compliance",
        "vital signs monitoring", "medication administration",
        "patient assessment", "care coordination", "cpr certified",
        "nursing", "diagnosis", "treatment planning",
    ],
    "Education & Training": [
        "curriculum development", "lesson planning", "classroom management",
        "instructional design", "student assessment", "tutoring",
        "e-learning", "public speaking", "training delivery",
    ],
    "Hospitality & Customer Service": [
        "customer service", "guest relations", "reservation management",
        "food and beverage service", "hospitality management",
        "event planning", "front desk operations", "complaint resolution",
        "upselling", "menu planning", "catering",
    ],
    "Manufacturing & Engineering": [
        "cad", "autocad", "solidworks", "quality control", "lean manufacturing",
        "six sigma", "process engineering", "root cause analysis",
        "preventive maintenance", "production planning", "safety compliance",
        "osha compliance", "blueprint reading",
    ],
    "Administrative": [
        "data entry", "scheduling", "calendar management", "office administration",
        "records management", "correspondence", "filing", "transcription",
        "travel coordination", "front office management",
    ],
    "Soft Skills": [
        "communication", "leadership", "problem solving", "teamwork",
        "time management", "critical thinking", "adaptability",
        "collaboration", "decision making", "attention to detail",
        "multitasking", "creativity", "interpersonal skills",
        "conflict management", "organizational skills",
    ],
}


def _flatten_skills() -> List[str]:
    """Return a flat, de-duplicated list of all known skills."""
    seen: Set[str] = set()
    flat: List[str] = []
    for skills in SKILLS_DB.values():
        for skill in skills:
            key = skill.lower().strip()
            if key not in seen:
                seen.add(key)
                flat.append(key)
    # Longer phrases first so multi-word skills are matched before
    # any shorter skill that might be a substring of them.
    flat.sort(key=len, reverse=True)
    return flat


ALL_SKILLS: List[str] = _flatten_skills()


def _build_pattern(skill: str) -> re.Pattern:
    """Build a case-insensitive, word-boundary-safe regex for a skill phrase."""
    escaped = re.escape(skill)
    # Allow '.', '+', '#' inside tech terms (e.g. c++, node.js) by using
    # lookaround boundaries instead of strict \b which can fail on symbols.
    pattern = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"
    return re.compile(pattern, re.IGNORECASE)


_SKILL_PATTERNS = {skill: _build_pattern(skill) for skill in ALL_SKILLS}


def extract_skills(text: str) -> List[str]:
    """
    Extract known skills mentioned in the given text.

    Args:
        text: Raw or cleaned text (resume text or job description text).

    Returns:
        Sorted list of unique matched skills (lowercase, as defined in SKILLS_DB).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    lowered = text.lower()
    found = set()
    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(lowered):
            found.add(skill)

    return sorted(found)


def compare_skills(resume_text: str, jd_text: str) -> Dict[str, object]:
    """
    Compare skills present in a resume against skills required by a job description.

    Args:
        resume_text: Text extracted from the candidate's resume.
        jd_text: Text of the target job description.

    Returns:
        A dictionary with:
            - "resume_skills": list of skills found in the resume
            - "jd_skills": list of skills found in the job description
            - "matching_skills": skills present in both
            - "missing_skills": skills required by JD but absent from resume
            - "match_percentage": float percentage of JD skills covered by resume
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    matching_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    if jd_skills:
        match_percentage = round((len(matching_skills) / len(jd_skills)) * 100, 2)
    else:
        match_percentage = 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "match_percentage": match_percentage,
    }
