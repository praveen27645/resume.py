import os
import sys

# Allow running this file directly by adding project root to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analyzer import analyze_resume


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
