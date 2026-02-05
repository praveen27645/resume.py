from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import io
import csv
import uuid
from datetime import datetime, timezone
import re

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional at runtime
    fitz = None

from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze_resume, analyze_resume_full

app = FastAPI(title="AI-Powered Resume Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    suggested = result.get("suggested", []) if result else []
    bullets = result.get("bullets", []) if result else []

    missing_list = "".join(f"<li>{w}</li>" for w in missing) or "<li>None</li>"
    suggested_list = "".join(f"<li>{w}</li>" for w in suggested) or "<li>None</li>"
    bullets_list = "".join(f"<li>{w}</li>" for w in bullets) or "<li>None</li>"
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

    script = """
        <script>
          const resumeEl = document.querySelector('textarea[name="resume_text"]');
          const jdEl = document.querySelector('textarea[name="job_description"]');
          const resultCard = document.getElementById('result-card');
          const scoreText = document.getElementById('score-text');
          const missingList = document.getElementById('missing-list');
          const suggestedList = document.getElementById('suggested-list');
          const bulletsList = document.getElementById('bullets-list');
          const downloadJson = document.getElementById('download-json');
          const downloadCsv = document.getElementById('download-csv');

          let timer = null;
          function debounceAnalyze() {
            clearTimeout(timer);
            timer = setTimeout(runAnalyze, 600);
          }

          async function runAnalyze() {
            const resume = resumeEl.value.trim();
            const jd = jdEl.value.trim();
            if (!resume || !jd) {
              resultCard.style.display = 'none';
              return;
            }
            try {
              const resp = await fetch('/api/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({resume_text: resume, job_description: jd})
              });
              const data = await resp.json();
              if (!resp.ok) {
                resultCard.style.display = 'none';
                return;
              }
              scoreText.innerHTML = `Score: <strong>${data.score}%</strong>`;
              missingList.innerHTML = (data.missing.length ? data.missing : ['None'])
                .map(w => `<li>${w}</li>`).join('');
              suggestedList.innerHTML = (data.suggested_keywords.length ? data.suggested_keywords : ['None'])
                .map(w => `<li>${w}</li>`).join('');
              bulletsList.innerHTML = (data.suggested_bullets.length ? data.suggested_bullets : ['None'])
                .map(w => `<li>${w}</li>`).join('');
              resultCard.style.display = 'block';
            } catch (e) {
              resultCard.style.display = 'none';
            }
          }

          resumeEl.addEventListener('input', debounceAnalyze);
          jdEl.addEventListener('input', debounceAnalyze);

          async function fetchReport() {
            const resume = resumeEl.value.trim();
            const jd = jdEl.value.trim();
            if (!resume || !jd) return null;
            const resp = await fetch('/api/report', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({resume_text: resume, job_description: jd})
            });
            if (!resp.ok) return null;
            return await resp.json();
          }

          function downloadBlob(blob, filename) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
          }

          downloadJson.addEventListener('click', async () => {
            const report = await fetchReport();
            if (!report) return;
            const blob = new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'});
            downloadBlob(blob, `resume_report_${report.id}.json`);
          });

          downloadCsv.addEventListener('click', async () => {
            const resume = resumeEl.value.trim();
            const jd = jdEl.value.trim();
            if (!resume || !jd) return;
            const resp = await fetch('/api/missing.csv', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({resume_text: resume, job_description: jd})
            });
            if (!resp.ok) return;
            const blob = await resp.blob();
            downloadBlob(blob, 'missing_keywords.csv');
          });
        </script>
    """

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
          .ghost {{
            background: transparent;
            color: var(--accent);
            border: 1px solid var(--accent);
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
              <form id="analyze-form" action="/analyze" method="post" enctype="multipart/form-data">
                <label>Resume PDF (optional)</label><br/>
                <input type="file" name="resume_file" accept=".pdf,.txt"/><br/><br/>

                <label>Resume Text</label>
                <textarea name="resume_text" placeholder="Paste resume text here...">{resume_text}</textarea>

                <label>Job Description</label>
                <textarea name="job_description" placeholder="Paste job description here...">{job_description}</textarea>

                <div class="row">
                  <button type="submit">Analyze</button>
                  <button type="button" id="download-json" class="ghost">Download JSON</button>
                  <button type="button" id="download-csv" class="ghost">Download CSV</button>
                </div>
              </form>
            </section>
            <section class="card result" id="result-card" style="display:{'block' if result else 'none'};">
              <h2>ATS Result</h2>
              <div class="score" id="score-text">Score: <strong>{score}%</strong></div>
              <h3>Missing Keywords</h3>
              <ul id="missing-list">{missing_list}</ul>
              <h3>Suggested Keywords to Add</h3>
              <ul id="suggested-list">{suggested_list}</ul>
              <h3>Suggested Resume Bullets</h3>
              <ul id="bullets-list">{bullets_list}</ul>
              <p class="hint">Tip: add relevant missing keywords naturally in your skills or experience.</p>
            </section>
          </div>
        </div>
        {script}
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
        full = analyze_resume_full(resume_text, job_description)
        return _render_page(
            result={
                "score": full["score"],
                "missing": full["missing_keywords"],
                "suggested": full["suggested_keywords"],
                "bullets": full["suggested_bullets"],
            },
            resume_text=resume_text,
            job_description=job_description,
        )
    except Exception as exc:
        return _render_page(
            error=str(exc),
            resume_text=resume_text,
            job_description=job_description,
        )


@app.post("/api/analyze")
def api_analyze(payload: dict):
    resume_text = (payload.get("resume_text") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    full = analyze_resume_full(resume_text, job_description)
    return {
        "score": full["score"],
        "missing": full["missing_keywords"],
        "suggested_keywords": full["suggested_keywords"],
        "suggested_bullets": full["suggested_bullets"],
    }


@app.post("/api/report")
def api_report(payload: dict):
    resume_text = (payload.get("resume_text") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    full = analyze_resume_full(resume_text, job_description)
    report = {
        "id": uuid.uuid4().hex[:10],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score": full["score"],
        "missing_keywords": full["missing_keywords"],
        "suggested_keywords": full["suggested_keywords"],
        "suggested_bullets": full["suggested_bullets"],
    }
    return JSONResponse(report)


@app.post("/api/missing.csv")
def api_missing_csv(payload: dict):
    resume_text = (payload.get("resume_text") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    full = analyze_resume_full(resume_text, job_description)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["keyword"])
    for kw in full["missing_keywords"]:
        writer.writerow([kw])

    output.seek(0)
    filename = "missing_keywords.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(output, media_type="text/csv", headers=headers)
