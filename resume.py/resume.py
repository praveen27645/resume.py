import argparse
import json
import os
import sys

# Allow running this file directly by adding project root to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analyzer import analyze_resume, analyze_resume_full, load_config


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _format_text(result: dict) -> str:
    lines = [
        "=== AI-Powered Resume Analyzer ===",
        "",
        f"ATS Score: {result['score']}%",
        f"Missing Keywords: {', '.join(result['missing_keywords']) or 'None'}",
        "",
        "Suggested Keywords:",
    ]
    lines.extend(f"- {kw}" for kw in (result["suggested_keywords"] or ["None"]))
    lines.append("")
    lines.append("Suggested Resume Bullets:")
    lines.extend(f"- {b}" for b in (result["suggested_bullets"] or ["None"]))
    lines.append("")
    lines.append("Breakdown:")
    for key, value in result["breakdown"].items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}%")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-Powered Resume Analyzer CLI")
    parser.add_argument("--resume-file", help="Path to resume text file")
    parser.add_argument("--resume-text", help="Resume text input")
    parser.add_argument("--jd-file", help="Path to job description text file")
    parser.add_argument("--job-description", help="Job description text input")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--output", help="Write output to a file instead of stdout")
    parser.add_argument("--config", help="Path to JSON config file")
    args = parser.parse_args()

    if not (args.resume_file or args.resume_text):
        print("=== AI-Powered Resume Analyzer ===\n")
        resume = input("Enter Resume Text:\n").strip()
        print("\n")
        job_desc = input("Enter Job Description:\n").strip()
        try:
            score, missing = analyze_resume(resume, job_desc, config=load_config(args.config))
        except ValueError as exc:
            print(f"\nError: {exc}")
            return 1
        print("\n--- Analysis Result ---")
        print(f"ATS Score: {score}%")
        if missing:
            print("Missing Keywords:", ", ".join(missing))
            print("Suggestion: Add missing keywords to improve ATS compatibility.")
        else:
            print("Excellent match! No major keywords missing.")
        return 0

    resume_text = ""
    if args.resume_file:
        resume_text = _read_text(args.resume_file)
    if args.resume_text:
        resume_text = f"{resume_text}\n{args.resume_text}".strip()

    jd_text = ""
    if args.jd_file:
        jd_text = _read_text(args.jd_file)
    if args.job_description:
        jd_text = f"{jd_text}\n{args.job_description}".strip()

    try:
        result = analyze_resume_full(
            resume_text,
            jd_text,
            config=load_config(args.config),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if args.format == "json":
        output = json.dumps(result, indent=2)
    else:
        output = _format_text(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
