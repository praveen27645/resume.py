# API Reference

## `POST /api/analyze`
Request body:
```
{
  "resume_text": "string",
  "job_description": "string"
}
```

Response:
```
{
  "score": 82.5,
  "missing": ["docker", "aws"],
  "suggested_keywords": ["docker", "aws"],
  "suggested_bullets": ["..."],
  "breakdown": {
    "tfidf_similarity": 70.5,
    "semantic_similarity": 84.1,
    "keyword_match_rate": 62.0,
    "skills_match_rate": 40.0
  }
}
```

## `POST /api/report`
Returns a report with a generated `id` and `created_at` timestamp.

## `POST /api/missing.csv`
Returns a CSV file with the missing keywords.

## `GET /api/sample`
Returns sample resume and job description text.
