import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize(text):
    # Keep only letters/numbers, collapse whitespace
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(tokens)


def analyze_resume(resume_text, job_description):
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if not resume_text:
        raise ValueError("Resume text is required.")
    if not job_description:
        raise ValueError("Job description is required.")

    resume_clean = _normalize(resume_text)
    jd_clean = _normalize(job_description)

    # Convert texts into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])

    # Calculate cosine similarity (ATS score)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    ats_score = round(similarity * 100, 2)

    # Identify missing keywords
    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())
    missing_keywords = sorted(jd_words - resume_words)

    return ats_score, missing_keywords[:10]


if __name__ == "__main__":
    print("=== AI-Powered Resume Analyzer ===\n")

    resume = input("Enter Resume Text:\n").strip()
    print("\n")
    job_desc = input("Enter Job Description:\n").strip()

    try:
        score, missing = analyze_resume(resume, job_desc)
    except ValueError as exc:
        print(f"\nError: {exc}")
        raise SystemExit(1)

    print("\n--- Analysis Result ---")
    print(f"ATS Score: {score}%")

    if missing:
        print("Missing Keywords:", ", ".join(missing))
        print("Suggestion: Add missing keywords to improve ATS compatibility.")
    else:
        print("Excellent match! No major keywords missing.")
