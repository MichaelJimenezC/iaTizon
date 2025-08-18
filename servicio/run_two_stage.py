# servicio/run_two_stage.py
import os, sys, json, argparse, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

# === usa tu propia ResNet y normalización ===
from src.model import ResNet18
from src.data import MEAN, STD

# ---------- util de logging a stderr ----------
def log(msg):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()

def die_json(error_msg, **extra):
    out = {"ok": False, "error": error_msg}
    out.update(extra)
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(1)

def ok_json(payload):
    payload.setdefault("ok", True)
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(0)

# ---------- paths ----------
# Usa tu ruta absoluta actual
MODELS_BASE = Path("/Users/user1/Documents/GitHub/iaTizon/servicio/models").resolve()
S1 = MODELS_BASE / "stage1"
S2 = MODELS_BASE / "stage2"

S1_PTH = S1 / "stage1_maize_not_v3.pth"
S2_PTH = S2 / "stage2_maize_tizon.pth"

def load_meta(dir_path: Path):
    cj = dir_path / "classes.json"
    if not cj.exists():
        die_json(f"classes.json no existe en {dir_path}", where=str(dir_path))
    try:
        classes = json.loads(cj.read_text(encoding="utf-8"))
    except Exception as e:
        die_json("No pude leer classes.json", where=str(dir_path), exc=str(e))

    tau_file = dir_path / "tau.txt"
    tau = None
    if tau_file.exists():
        try:
            tau = float(tau_file.read_text().strip())
        except Exception as e:
            log(f"[WARN] tau.txt malformado en {dir_path}: {e}")
            tau = None
    return classes, tau

CLASSES1, TAU1 = load_meta(S1)   # ['maize', 'not_maize'], tau≈0.272
CLASSES2, TAU2 = load_meta(S2)   # ['healthy','tizon_foliar'], tau≈0.395

# ---------- device ----------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
log(f"[device] {DEVICE}")

# ---------- tfm ----------
TFM = transforms.Compose([
    transforms.Resize(int(320*1.14)),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_model(pth: Path, ncls: int):
    if not pth.exists():
        die_json("Checkpoint .pth no existe", path=str(pth))
    try:
        m = ResNet18(num_classes=ncls).to(DEVICE).eval()
        state = torch.load(str(pth), map_location=DEVICE)
        state = state.get("model", state)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            log(f"[WARN] missing_keys={missing}, unexpected_keys={unexpected}")
        return m
    except Exception as e:
        die_json("Error cargando modelo", path=str(pth), exc=str(e))

M1 = load_model(S1_PTH, len(CLASSES1))
M2 = load_model(S2_PTH, len(CLASSES2))
log(f"[stage1] classes={CLASSES1} tau={TAU1}")
log(f"[stage2] classes={CLASSES2} tau={TAU2}")

@torch.no_grad()
def infer_probs(model, pil_img):
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    x = TFM(pil_img).unsqueeze(0).to(DEVICE, non_blocking=True)
    y = model(x)
    p = torch.softmax(y, 1)[0].cpu().tolist()
    return p

def apply_tau_binary(probs, classes, tau, pos_class):
    if tau is None or len(classes) != 2 or pos_class not in classes:
        idx = int(torch.tensor(probs).argmax().item())
        return classes[idx]
    pos_idx = classes.index(pos_class)
    return pos_class if float(probs[pos_idx]) >= float(tau) else classes[1 - pos_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    args = ap.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        die_json("Imagen no existe", path=str(img_path))

    try:
        img = Image.open(str(img_path))
    except Exception as e:
        die_json("No pude abrir la imagen", path=str(img_path), exc=str(e))

    # ---- STAGE 1: maize vs not_maize ----
    try:
        p1 = infer_probs(M1, img)
    except Exception as e:
        die_json("Fallo inferencia stage1", exc=str(e), tb=traceback.format_exc())

    pred1 = apply_tau_binary(p1, CLASSES1, TAU1, pos_class="maize")
    stage1 = {"pred": pred1, "probs": dict(zip(CLASSES1, map(float, p1))), "tau": TAU1}

    if pred1 != "maize":
        ok_json({"final": "not_maize", "stage1": stage1})

    # ---- STAGE 2: healthy vs tizon_foliar ----
    try:
        p2 = infer_probs(M2, img)
    except Exception as e:
        die_json("Fallo inferencia stage2", exc=str(e), tb=traceback.format_exc())

    pred2 = apply_tau_binary(p2, CLASSES2, TAU2, pos_class="tizon_foliar")
    stage2 = {"pred": pred2, "probs": dict(zip(CLASSES2, map(float, p2))), "tau": TAU2}

    ok_json({"final": pred2, "stage1": stage1, "stage2": stage2})

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        die_json("Excepción no manejada", exc=str(e), tb=traceback.format_exc())
