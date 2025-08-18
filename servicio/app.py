from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os
import subprocess
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # o ["*"] mientras desarrollas
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)
# Carpeta temporal donde se guardan imágenes subidas
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Guardar imagen temporal
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ejecutar tu servicio de predicción (run_two_stage.py)
    result = subprocess.run(
        ["python", "run_two_stage.py", "--img", temp_path],
        capture_output=True, text=True
    )

    # Leer salida
    output = result.stdout

    return {"output": output}
