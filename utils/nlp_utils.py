# import spacy
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # Load models globally (so they load once)
# nlp = spacy.load("en_core_web_sm")
# bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# def compute_tfidf_similarity(jd_text, resume_text):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
#     return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]

# def compute_bert_similarity(jd_text, resume_text):
#     jd_embedding = bert_model.encode([jd_text], convert_to_tensor=True).cpu().numpy()
#     resume_embedding = bert_model.encode([resume_text], convert_to_tensor=True).cpu().numpy()
#     return cosine_similarity(jd_embedding, resume_embedding).flatten()[0]
# import spacy
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import re
# from collections import Counter

# # ---------------- Load NLP Models ----------------
# nlp = spacy.load("en_core_web_sm")
# bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ---------------- TF-IDF Similarity ----------------
# def compute_tfidf_similarity(jd_text: str, resume_text: str) -> float:
#     """
#     Compute TF-IDF cosine similarity between job description and resume.
#     """
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
#     sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
#     return sim

# # ---------------- BERT Similarity ----------------
# def compute_bert_similarity(jd_text: str, resume_text: str) -> float:
#     """
#     Compute BERT-based semantic similarity between job description and resume.
#     """
#     jd_emb = bert_model.encode([jd_text], convert_to_tensor=True).cpu().numpy()
#     resume_emb = bert_model.encode([resume_text], convert_to_tensor=True).cpu().numpy()
#     sim = cosine_similarity(jd_emb, resume_emb).flatten()[0]
#     return sim

# # ---------------- Combined Similarity ----------------
# def compute_combined_similarity(jd_text: str, resume_text: str, weights=(0.5, 0.5)) -> float:
#     """
#     Weighted combination of TF-IDF and BERT similarity.
#     weights: (tfidf_weight, bert_weight)
#     """
#     tfidf_sim = compute_tfidf_similarity(jd_text, resume_text)
#     bert_sim = compute_bert_similarity(jd_text, resume_text)
#     combined = tfidf_sim * weights[0] + bert_sim * weights[1]
#     return combined

# # ---------------- Skill / Keyword Extraction ----------------
# def extract_skills(resume_text: str, skills_list: list) -> list:
#     """
#     Extract relevant skills from a resume text.
#     Returns a list of found skills.
#     """
#     resume_doc = nlp(resume_text.lower())
#     found_skills = [skill for skill in skills_list if skill.lower() in resume_doc.text]
#     return found_skills

# # ---------------- Resume Shortlisting ----------------
# def shortlist_resumes(similarities: list, threshold: float = 0.5) -> list:
#     """
#     Shortlist resumes with similarity above a threshold.
#     """
#     shortlisted_indices = [i for i, score in enumerate(similarities) if score >= threshold]
#     return shortlisted_indices
# # Add this function at the bottom


# import re
# from collections import Counter
# import spacy

# nlp = spacy.load("en_core_web_sm")

# def extract_skills(resume_text, skills_list=None):
#     """
#     Extract skills from resume text.
#     If skills_list is provided, match against it (case-insensitive).
#     Otherwise, extract top frequent keywords.
#     """
#     text_lower = resume_text.lower()
    
#     if skills_list:
#         # NLP tokenization for more accurate matching
#         doc = nlp(text_lower)
#         found_skills = [skill for skill in skills_list if skill.lower() in text_lower]
#         return list(set(found_skills))
    
#     # If no predefined list, fall back to most common keywords
#     words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
#     freq = Counter(words)
    
#     # Remove generic stopwords
#     stopwords = set(nlp.Defaults.stop_words)
#     most_common = [word for word, _ in freq.most_common(15) if word not in stopwords]
    
#     return most_common
"""
nlp_utils.py
Advanced NLP utilities for AI Resume Screening & Ranking
--------------------------------------------------------
Features:
- TF-IDF, BERT, and hybrid similarity scoring
- Fuzzy skill extraction with CSV-based dictionary
- Skill categorization and weighted scoring
- Keyword frequency & shared term analysis
- Named entity extraction
- Resume shortlisting logic
"""

import re
import logging
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# Optional: for wordclouds/plots if app requests them
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    WordCloud = None
    plt = None

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load Models ----------------
logger.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser"])

logger.info("Loading SentenceTransformer model...")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Load Skills Dictionary ----------------
try:
    skills_df = pd.read_csv("data/skills_master.csv")
    SKILL_LIST = skills_df["Skill"].dropna().tolist()
    SKILL_CATEGORIES = dict(
        zip(
            skills_df["Skill"],
            skills_df.get("Category", ["Other"] * len(skills_df))
        )
    )
    logger.info(f"Loaded {len(SKILL_LIST)} skills from CSV.")
except FileNotFoundError:
    logger.warning("Skills CSV not found. Skill extraction will be limited.")
    SKILL_LIST = []
    SKILL_CATEGORIES = {}

# ---------------- Similarity Scoring ----------------
def compute_tfidf_similarity(jd_text: str, resume_text: str) -> float:
    """Compute TF-IDF cosine similarity between job description and resume."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0])


def compute_bert_similarity(jd_text: str, resume_text: str) -> float:
    """Compute BERT-based semantic similarity."""
    jd_emb = bert_model.encode([jd_text], convert_to_tensor=True).cpu().numpy()
    resume_emb = bert_model.encode([resume_text], convert_to_tensor=True).cpu().numpy()
    return float(cosine_similarity(jd_emb, resume_emb).flatten()[0])


def compute_combined_similarity(
    jd_text: str,
    resume_text: str,
    weights: Tuple[float, float] = (0.5, 0.5)
) -> float:
    """Weighted combination of TF-IDF and BERT similarity."""
    tfidf_sim = compute_tfidf_similarity(jd_text, resume_text)
    bert_sim = compute_bert_similarity(jd_text, resume_text)
    return round(tfidf_sim * weights[0] + bert_sim * weights[1], 4)

# ---------------- Skill Extraction ----------------
def extract_skills(resume_text: str, threshold: int = 85) -> List[str]:
    """
    Extract skills from resume text using fuzzy matching against SKILL_LIST.
    threshold: minimum fuzzy match score (0-100)
    """
    if not SKILL_LIST:
        return []

    text_lower = resume_text.lower()
    found_skills = set()

    for skill in SKILL_LIST:
        skill_lower = skill.lower()
        if re.search(rf"\b{re.escape(skill_lower)}\b", text_lower):
            found_skills.add(skill)
        else:
            if fuzz.partial_ratio(skill_lower, text_lower) >= threshold:
                found_skills.add(skill)

    return sorted(found_skills)


def extract_skills_with_categories(resume_text: str) -> Dict[str, List[str]]:
    """Extract skills and group them by category."""
    skills_found = extract_skills(resume_text)
    categorized = {}
    for skill in skills_found:
        category = SKILL_CATEGORIES.get(skill, "Other")
        categorized.setdefault(category, []).append(skill)
    return categorized


def skill_match_score(resume_text: str, jd_text: str) -> float:
    """Compute skill match percentage between resume and job description."""
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    return round(len(resume_skills & jd_skills) / len(jd_skills), 4) if jd_skills else 0.0

# ---------------- Keyword & Shared Terms ----------------
def keyword_frequency(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """Return top N keywords (excluding stopwords)."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    stopwords = set(nlp.Defaults.stop_words)
    filtered = [w for w in words if w not in stopwords]
    return Counter(filtered).most_common(top_n)


def shared_terms(jd_text: str, resume_text: str, top_n: int = 50) -> Counter:
    """Return terms that appear in BOTH JD and resume."""
    jd_tokens = set(_basic_tokens(jd_text))
    res_tokens = _basic_tokens(resume_text)
    freq = Counter([t for t in res_tokens if t in jd_tokens])
    return Counter(dict(freq.most_common(top_n)))


def _basic_tokens(text: str) -> List[str]:
    """Lowercase + simple tokenization, strips non-letters."""
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    stopwords = set(nlp.Defaults.stop_words)
    return [t for t in tokens if t not in stopwords]

# ---------------- Optional WordCloud ----------------
def make_overlap_wordcloud(jd_text: str, resume_text: str, max_words: int = 60):
    """
    Generate a matplotlib Figure with a wordcloud of shared JD–resume terms.
    Returns: matplotlib.figure.Figure
    """
    if WordCloud is None or plt is None:
        logger.error("WordCloud/matplotlib not installed.")
        return None

    terms = shared_terms(jd_text, resume_text, top_n=max_words)
    if not terms:
        fig = plt.figure(figsize=(4, 2))
        plt.axis("off")
        plt.text(0.5, 0.5, "No shared keywords found", ha="center", va="center")
        return fig

    wc = WordCloud(width=800, height=400, background_color="white")
    wc = wc.generate_from_frequencies(terms)

    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig

# ---------------- Entity Extraction ----------------
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities (ORG, GPE, DATE, PERSON, etc.) from text."""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, set()).add(ent.text)
    return {k: sorted(v) for k, v in entities.items()}

# ---------------- Resume Shortlisting ----------------
def shortlist_resumes(similarities: List[float], threshold: float = 0.5) -> List[int]:
    """Return indices of resumes with similarity >= threshold."""
    return [i for i, score in enumerate(similarities) if score >= threshold]
