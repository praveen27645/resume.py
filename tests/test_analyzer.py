from analyzer import AnalyzerConfig, analyze_resume_full


def test_analyze_resume_full_basic():
    resume = "Python developer with SQL and FastAPI experience."
    jd = "Looking for a Python engineer with SQL, Docker, and AWS experience."
    config = AnalyzerConfig(top_n=10, weights=(0.4, 0.4, 0.2))

    result = analyze_resume_full(resume, jd, config=config)

    assert 0 <= result["score"] <= 100
    assert "missing_keywords" in result
    assert "suggested_keywords" in result
    assert "suggested_bullets" in result
    assert "breakdown" in result
    assert isinstance(result["breakdown"], dict)

    missing = result["missing_keywords"]
    assert any(keyword in missing for keyword in ["docker", "aws"])
