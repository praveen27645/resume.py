from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
import re

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional at runtime
    fitz = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="AI-Powered Resume Analyzer")


def _normalize(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(tokens)


def analyze_resume(resume_text: str, job_description: str):
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if not resume_text:
        raise ValueError("Resume text is required (paste it or upload a PDF).")
    if not job_description:
        raise ValueError("Job description is required.")

    resume_clean = _normalize(resume_text)
    jd_clean = _normalize(job_description)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    ats_score = round(similarity * 100, 2)

    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())
    missing_keywords = sorted(jd_words - resume_words)

    return ats_score, missing_keywords[:10]


def _extract_text_from_upload(upload: UploadFile) -> str:
    filename = (upload.filename or "").lower()
    data = upload.file.read()

    if filename.endswith(".pdf"):
        if fitz is None:
            raise ValueError("PDF support requires PyMuPDF. Install pymupdf.")
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    # treat everything else as text
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def _render_page(result=None, error=None, resume_text="", job_description=""):
    score = result["score"] if result else ""
    missing = result["missing"] if result else []

    missing_list = "".join(f"<li>{w}</li>" for w in missing) or "<li>None</li>"
    result_block = ""
    if result:
        result_block = f"""
        <section class="card result">
          <h2>ATS Result</h2>
          <div class="score">Score: <strong>{score}%</strong></div>
          <h3>Missing Keywords</h3>
          <ul>{missing_list}</ul>
          <p class="hint">Tip: add relevant missing keywords naturally in your skills or experience.</p>
        </section>
        """

    error_block = f'<div class="error">{error}</div>' if error else ""

    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>AI Resume Analyzer</title>
        <style>
          :root {{
            --ink: #0f172a;
            --muted: #475569;
            --accent: #0ea5e9;
            --bg: #f7f7fb;
            --card: #ffffff;
            --shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
          }}
          body {{
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at top left, #e0f2fe, #f7f7fb 55%);
            margin: 0;
            padding: 32px 16px;
          }}
          .container {{
            max-width: 980px;
            margin: 0 auto;
          }}
          header {{
            margin-bottom: 24px;
          }}
          h1 {{
            margin: 0 0 6px;
            font-size: 32px;
          }}
          .subtitle {{
            color: var(--muted);
          }}
          .grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
          }}
          .card {{
            background: var(--card);
            border-radius: 16px;
            padding: 18px;
            box-shadow: var(--shadow);
          }}
          label {{
            font-weight: 600;
          }}
          textarea {{
            width: 100%;
            min-height: 160px;
            padding: 10px;
            margin-top: 6px;
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            font-size: 14px;
          }}
          input[type="file"] {{
            margin-top: 6px;
          }}
          button {{
            background: var(--accent);
            color: white;
            border: 0;
            padding: 10px 16px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
          }}
          .row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
          }}
          .error {{
            background: #fee2e2;
            color: #991b1b;
            padding: 10px 12px;
            border-radius: 10px;
            margin-bottom: 12px;
          }}
          .score {{
            font-size: 20px;
            margin: 8px 0 12px;
          }}
          .hint {{
            color: var(--muted);
            font-size: 14px;
          }}
          @media (min-width: 860px) {{
            .grid {{
              grid-template-columns: 2fr 1fr;
              align-items: start;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <header>
            <h1>AI-Powered Resume Analyzer</h1>
            <div class="subtitle">Upload a resume or paste text, add a job description, and get an ATS-style match score.</div>
          </header>
          {error_block}
          <div class="grid">
            <section class="card">
              <form action="/analyze" method="post" enctype="multipart/form-data">
                <label>Resume PDF (optional)</label><br/>
                <input type="file" name="resume_file" accept=".pdf,.txt"/><br/><br/>

                <label>Resume Text</label>
                <textarea name="resume_text" placeholder="Paste resume text here...">{resume_text}</textarea>

                <label>Job Description</label>
                <textarea name="job_description" placeholder="Paste job description here...">{job_description}</textarea>

                <div class="row">
                  <button type="submit">Analyze</button>
                </div>
              </form>
            </section>
            {result_block}
          </div>
        </div>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return _render_page()


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    resume_text: str = Form(""),
    job_description: str = Form(""),
    resume_file: UploadFile | None = File(None),
):
    try:
        if resume_file and resume_file.filename:
            resume_text = _extract_text_from_upload(resume_file)
        score, missing = analyze_resume(resume_text, job_description)
        return _render_page(
            result={"score": score, "missing": missing},
            resume_text=resume_text,
            job_description=job_description,
        )
    except Exception as exc:
        return _render_page(
            error=str(exc),
            resume_text=resume_text,
            job_description=job_description,
        )
