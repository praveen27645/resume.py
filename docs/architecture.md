# Architecture

## Overview
The application consists of a FastAPI web service and a reusable analyzer module.

## Components
- `main.py` provides the web UI and API endpoints.
- `analyzer.py` performs NLP normalization, scoring, and keyword analysis.
- `resume.py/resume.py` offers a CLI interface.

## Data Flow
1. User submits resume text + job description.
2. Analyzer normalizes and lemmatizes text.
3. The system computes TF-IDF, semantic similarity, and keyword match.
4. Results are returned with missing keywords and suggested bullets.
