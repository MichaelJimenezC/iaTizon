# servicio/run_two_stage.py
import os, sys, json, argparse, traceback
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

# === tu ResNet y normalización ===
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
MODELS_BASE = (Path(__file__).resolve().parent / "models").resolve()
BUNDLE_PTH  = MODELS_BASE / "maize_two_stage.pth"

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

def load_bundle(pth: Path):
    if not pth.exists():
        die_json("Bundle .pth no existe", path=str(pth))
    try:
        ckpt = torch.load(str(pth), map_location=DEVICE)

        # metadatos
        classes1 = ckpt["classes1"]; tau1 = ckpt.get("tau1")
        classes2 = ckpt["classes2"]; tau2 = ckpt.get("tau2")

        # modelos
        m1 = ResNet18(num_classes=len(classes1)).to(DEVICE).eval()
        m2 = ResNet18(num_classes=len(classes2)).to(DEVICE).eval()

        st1 = ckpt["stage1"]
        st2 = ckpt["stage2"]
        miss1, unexp1 = m1.load_state_dict(st1, strict=False)
        miss2, unexp2 = m2.load_state_dict(st2, strict=False)
        if miss1 or unexp1: log(f"[WARN stage1] missing={miss1} unexpected={unexp1}")
        if miss2 or unexp2: log(f"[WARN stage2] missing={miss2} unexpected={unexp2}")

        return m1, m2, classes1, tau1, classes2, tau2
    except Exception as e:
        die_json("Error leyendo bundle", path=str(pth), exc=str(e), tb=traceback.format_exc())

M1, M2, CLASSES1, TAU1, CLASSES2, TAU2 = load_bundle(BUNDLE_PTH)
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
