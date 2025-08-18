# servicio/app.py (fragmento)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid, os, shutil, subprocess, sys, json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPT   = BASE_DIR / "run_two_stage.py"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000","http://localhost:5173","http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    tmp = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cmd = [sys.executable, str(SCRIPT), "--img", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))

    # Limpia si quieres:
    # try: tmp.unlink(missing_ok=True)
    # except: pass

    if proc.returncode != 0:
        return {"ok": False, "error": "runner_failed", "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr}

    out = proc.stdout.strip()
    try:
        parsed = json.loads(out)
        return parsed
    except Exception:
        # si stdout no es JSON, te lo regresamos crudo con stderr
        return {"ok": False, "error": "invalid_json", "stdout": out, "stderr": proc.stderr}
