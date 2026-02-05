from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import io
import csv
import uuid
from datetime import datetime, timezone
import re
import os

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional at runtime
    fitz = None

from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze_resume, analyze_resume_full, load_config

app = FastAPI(title="AI-Powered Resume Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
ANALYZER_CONFIG = load_config()


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


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return ""


def _load_samples() -> tuple[str, str]:
    resume_path = os.path.join(EXAMPLES_DIR, "resume_sample.txt")
    jd_path = os.path.join(EXAMPLES_DIR, "job_description_sample.txt")
    resume_text = _read_text_file(resume_path)
    jd_text = _read_text_file(jd_path)
    if not resume_text:
        resume_text = "Sample resume text not found. Add examples/resume_sample.txt."
    if not jd_text:
        jd_text = "Sample job description text not found. Add examples/job_description_sample.txt."
    return resume_text, jd_text


def _render_page(result=None, error=None, resume_text="", job_description=""):
    score = result["score"] if result else ""
    missing = result["missing"] if result else []
    suggested = result.get("suggested", []) if result else []
    bullets = result.get("bullets", []) if result else []
    breakdown = result.get("breakdown", {}) if result else {}

    missing_list = "".join(f"<li>{w}</li>" for w in missing) or "<li>None</li>"
    suggested_list = "".join(f"<li>{w}</li>" for w in suggested) or "<li>None</li>"
    bullets_list = "".join(f"<li>{w}</li>" for w in bullets) or "<li>None</li>"
    breakdown_rows = ""
    if breakdown:
        for label, value in breakdown.items():
            safe_label = label.replace("_", " ").title()
            bar = f"""
            <div class="metric">
              <div class="metric-head">
                <span>{safe_label}</span>
                <span>{value}%</span>
              </div>
              <div class="meter"><span style="width:{value}%"></span></div>
            </div>
            """
            breakdown_rows += bar
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
          const breakdownList = document.getElementById('breakdown-list');
          const downloadJson = document.getElementById('download-json');
          const downloadCsv = document.getElementById('download-csv');
          const loadSample = document.getElementById('load-sample');
          const clearAll = document.getElementById('clear-all');

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
              breakdownList.innerHTML = Object.entries(data.breakdown || {})
                .map(([k, v]) => `
                  <div class="metric">
                    <div class="metric-head">
                      <span>${k.replace(/_/g, ' ')}</span>
                      <span>${v}%</span>
                    </div>
                    <div class="meter"><span style="width:${v}%"></span></div>
                  </div>
                `).join('');
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

          loadSample.addEventListener('click', async () => {
            const resp = await fetch('/api/sample');
            if (!resp.ok) return;
            const data = await resp.json();
            resumeEl.value = data.resume_text || '';
            jdEl.value = data.job_description || '';
            runAnalyze();
          });

          clearAll.addEventListener('click', () => {
            resumeEl.value = '';
            jdEl.value = '';
            resultCard.style.display = 'none';
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
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Code+Pro:wght@400;600&display=swap');
          :root {{
            --ink: #0f172a;
            --muted: #475569;
            --accent: #0f766e;
            --accent-2: #f97316;
            --bg: #f3f4f6;
            --card: #ffffff;
            --shadow: 0 24px 60px rgba(15, 23, 42, 0.16);
          }}
          body {{
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            color: var(--ink);
            background:
              radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 50%),
              radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 45%),
              linear-gradient(120deg, #f3f4f6 0%, #f8fafc 50%, #eef2ff 100%);
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
          .card.highlight {{
            border: 1px solid rgba(15, 118, 110, 0.1);
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
            font-family: "Source Code Pro", "Space Grotesk", monospace;
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
            transition: transform 0.15s ease;
          }}
          button:hover {{
            transform: translateY(-1px);
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
          .accent {{
            background: var(--accent-2);
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
          .metric {{
            margin-bottom: 12px;
          }}
          .metric-head {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: var(--muted);
            margin-bottom: 6px;
            text-transform: capitalize;
          }}
          .meter {{
            height: 8px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
          }}
          .meter span {{
            display: block;
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
          }}
          .pill {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.12);
            color: var(--accent);
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
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
            <div class="subtitle">Upload a resume or paste text, add a job description, and get an ATS-style match score with smart suggestions.</div>
            <div class="pill">Live keyword match</div>
            <div class="pill">ATS breakdown</div>
            <div class="pill">Exportable reports</div>
          </header>
          {error_block}
          <div class="grid">
            <section class="card highlight">
              <form id="analyze-form" action="/analyze" method="post" enctype="multipart/form-data">
                <label>Resume PDF (optional)</label><br/>
                <input type="file" name="resume_file" accept=".pdf,.txt"/><br/><br/>

                <label>Resume Text</label>
                <textarea name="resume_text" placeholder="Paste resume text here...">{resume_text}</textarea>

                <label>Job Description</label>
                <textarea name="job_description" placeholder="Paste job description here...">{job_description}</textarea>

                <div class="row">
                  <button type="submit">Analyze</button>
                  <button type="button" id="load-sample" class="ghost">Load Sample</button>
                  <button type="button" id="clear-all" class="ghost">Clear</button>
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
              <h3>Match Breakdown</h3>
              <div id="breakdown-list">{breakdown_rows}</div>
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
        full = analyze_resume_full(resume_text, job_description, config=ANALYZER_CONFIG)
        return _render_page(
            result={
                "score": full["score"],
                "missing": full["missing_keywords"],
                "suggested": full["suggested_keywords"],
                "bullets": full["suggested_bullets"],
                "breakdown": full["breakdown"],
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
    full = analyze_resume_full(resume_text, job_description, config=ANALYZER_CONFIG)
    return {
        "score": full["score"],
        "missing": full["missing_keywords"],
        "suggested_keywords": full["suggested_keywords"],
        "suggested_bullets": full["suggested_bullets"],
        "breakdown": full["breakdown"],
    }


@app.post("/api/report")
def api_report(payload: dict):
    resume_text = (payload.get("resume_text") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    full = analyze_resume_full(resume_text, job_description, config=ANALYZER_CONFIG)
    report = {
        "id": uuid.uuid4().hex[:10],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score": full["score"],
        "missing_keywords": full["missing_keywords"],
        "suggested_keywords": full["suggested_keywords"],
        "suggested_bullets": full["suggested_bullets"],
        "breakdown": full["breakdown"],
    }
    return JSONResponse(report)


@app.post("/api/missing.csv")
def api_missing_csv(payload: dict):
    resume_text = (payload.get("resume_text") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    full = analyze_resume_full(resume_text, job_description, config=ANALYZER_CONFIG)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["keyword"])
    for kw in full["missing_keywords"]:
        writer.writerow([kw])

    output.seek(0)
    filename = "missing_keywords.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(output, media_type="text/csv", headers=headers)


@app.get("/api/sample")
def api_sample():
    resume_text, job_description = _load_samples()
    return {"resume_text": resume_text, "job_description": job_description}
