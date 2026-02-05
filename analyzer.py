import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "js": "javascript",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "k8s": "kubernetes",
}


@dataclass(frozen=True)
class AnalyzerConfig:
    top_n: int = 30
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    synonyms: Mapping[str, str] = field(default_factory=lambda: dict(_SYNONYMS))


def _normalize_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    total = sum(weights)
    if total <= 0:
        return (0.4, 0.4, 0.2)
    return tuple(w / total for w in weights)


def load_config(path: str | None = None) -> AnalyzerConfig:
    config_path = path or os.getenv("RESUME_ANALYZER_CONFIG")
    if not config_path:
        return AnalyzerConfig()
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return AnalyzerConfig()

    try:
        top_n = int(data.get("top_n", 30))
    except Exception:
        top_n = 30
    top_n = max(5, top_n)
    raw_weights = data.get("weights", [0.4, 0.4, 0.2])
    if isinstance(raw_weights, list) and len(raw_weights) == 3:
        weights = _normalize_weights(tuple(float(w) for w in raw_weights))
    else:
        weights = (0.4, 0.4, 0.2)
    synonyms = dict(_SYNONYMS)
    for key, value in (data.get("synonyms") or {}).items():
        synonyms[str(key).lower()] = str(value).lower()
    return AnalyzerConfig(top_n=top_n, weights=weights, synonyms=synonyms)


@lru_cache(maxsize=1)
def _get_nlp():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_embedder():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _normalize(text: str, synonyms: Mapping[str, str]) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    expanded = []
    for tok in tokens:
        expanded.append(tok)
        if tok in synonyms:
            expanded.extend(synonyms[tok].split())
    return " ".join(expanded)


def _lemmatize(text: str, synonyms: Mapping[str, str]) -> str:
    nlp = _get_nlp()
    if nlp is None:
        return _normalize(text, synonyms)
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip()
        if lemma:
            tokens.append(lemma)
    return _normalize(" ".join(tokens), synonyms)


def _extract_skills_section(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^(skills?|technical skills?)\b", line.lower()):
            start_idx = i + 1
            break
    if start_idx == -1:
        return ""
    section = []
    for line in lines[start_idx:]:
        if not line:
            continue
        if re.match(r"^[A-Z][A-Za-z ]{2,}$", line) and len(section) > 2:
            break
        section.append(line)
        if len(section) > 8:
            break
    return " ".join(section)


def _keyword_list(job_description: str, top_n: int = 30):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    vectorizer.fit([job_description])
    return list(vectorizer.get_feature_names_out())


def _semantic_score(resume_text: str, job_description: str) -> float:
    embedder = _get_embedder()
    if embedder is None:
        return 0.0
    embeddings = embedder.encode([resume_text, job_description], normalize_embeddings=True)
    sim = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(sim, 1.0))


@lru_cache(maxsize=64)
def _jd_features(jd_clean: str, top_n: int):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    jd_vec = vectorizer.fit_transform([jd_clean])
    jd_keywords = _keyword_list(jd_clean, top_n=top_n)
    jd_embed = None
    embedder = _get_embedder()
    if embedder is not None:
        jd_embed = embedder.encode([jd_clean], normalize_embeddings=True)[0]
    return vectorizer, jd_vec, jd_keywords, jd_embed


def _suggest_bullets(missing_keywords: list[str]) -> list[str]:
    if not missing_keywords:
        return []
    top = missing_keywords[:5]
    bullets = [
        f"Integrated {top[0]} to improve performance and scalability.",
        f"Built and optimized workflows using {top[1] if len(top) > 1 else top[0]}.",
        f"Collaborated across teams to deliver features involving {top[2] if len(top) > 2 else top[0]}.",
        f"Applied best practices for {top[3] if len(top) > 3 else top[0]} in production systems.",
        f"Implemented monitoring and testing around {top[4] if len(top) > 4 else top[0]}.",
    ]
    return bullets[: min(3, len(bullets))]


def analyze_resume_full(
    resume_text: str,
    job_description: str,
    config: AnalyzerConfig | None = None,
):
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if not resume_text:
        raise ValueError("Resume text is required (paste it or upload a PDF).")
    if not job_description:
        raise ValueError("Job description is required.")

    cfg = config or AnalyzerConfig()
    resume_clean = _lemmatize(resume_text, cfg.synonyms)
    jd_clean = _lemmatize(job_description, cfg.synonyms)

    vectorizer, jd_vec, jd_keywords, jd_embed = _jd_features(jd_clean, cfg.top_n)
    resume_vec = vectorizer.transform([resume_clean])
    tfidf_sim = cosine_similarity(resume_vec, jd_vec)[0][0]

    if jd_embed is not None:
        embedder = _get_embedder()
        resume_embed = embedder.encode([resume_clean], normalize_embeddings=True)[0]
        sem_sim = float(np.dot(resume_embed, jd_embed))
    else:
        sem_sim = _semantic_score(resume_clean, jd_clean)
    resume_tokens = set(resume_clean.split())
    skills_text = _lemmatize(_extract_skills_section(resume_text), cfg.synonyms)
    skills_tokens = set(skills_text.split())

    if jd_keywords:
        matches = sum(1 for k in jd_keywords if k in resume_tokens)
        skills_matches = sum(1 for k in jd_keywords if k in skills_tokens)
        keyword_score = (matches + 0.5 * skills_matches) / len(jd_keywords)
    else:
        keyword_score = 0.0

    w_tfidf, w_sem, w_kw = cfg.weights
    final_score = (w_tfidf * tfidf_sim + w_sem * sem_sim + w_kw * keyword_score) * 100
    ats_score = round(final_score, 2)

    missing_keywords = sorted([k for k in jd_keywords if k not in resume_tokens])

    missing_top = missing_keywords[:10]
    suggested_keywords = missing_top
    suggested_bullets = _suggest_bullets(missing_top)

    match_rate = round((matches / len(jd_keywords)) * 100, 2) if jd_keywords else 0.0
    skills_rate = round((skills_matches / len(jd_keywords)) * 100, 2) if jd_keywords else 0.0
    breakdown = {
        "tfidf_similarity": round(tfidf_sim * 100, 2),
        "semantic_similarity": round(sem_sim * 100, 2),
        "keyword_match_rate": match_rate,
        "skills_match_rate": skills_rate,
    }

    return {
        "score": ats_score,
        "missing_keywords": missing_top,
        "suggested_keywords": suggested_keywords,
        "suggested_bullets": suggested_bullets,
        "breakdown": breakdown,
    }


def analyze_resume(resume_text: str, job_description: str, config: AnalyzerConfig | None = None):
    result = analyze_resume_full(resume_text, job_description, config=config)
    return result["score"], result["missing_keywords"]
