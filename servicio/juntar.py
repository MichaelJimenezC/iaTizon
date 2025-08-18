from pathlib import Path
import json, torch, sys


candidates = [
    Path(__file__).resolve().parent / "models",
    Path.cwd() / "models",
    Path.cwd() / "servicio" / "models",
]

MODELS_BASE = None
for p in candidates:
    if (p / "stage1").exists() and (p / "stage2").exists():
        MODELS_BASE = p.resolve()
        break

if MODELS_BASE is None:
    raise SystemExit(f"❌ No encuentro la carpeta 'models'. Probé: " +
                     " | ".join(str(p) for p in candidates))

print(f"✅ Usando MODELS_BASE = {MODELS_BASE}")

S1 = MODELS_BASE / "stage1"
S2 = MODELS_BASE / "stage2"

S1_PTH = S1 / "stage1_maize_not_v3.pth"
S2_PTH = S2 / "stage2_maize_tizon.pth"

def read_meta(dirp: Path):
    cj = dirp / "classes.json"
    if not cj.exists():
        raise SystemExit(f"❌ Falta {cj}")
    classes = json.loads(cj.read_text(encoding="utf-8"))
    tau = None
    tauf = dirp / "tau.txt"
    if tauf.exists():
        try:
            tau = float(tauf.read_text().strip())
        except Exception:
            tau = None
    return classes, tau

# lee metadatos
classes1, tau1 = read_meta(S1)   
classes2, tau2 = read_meta(S2)  

def load_state_dict(pth: Path):
    if not pth.exists():
        raise SystemExit(f"❌ Falta {pth}")
    state = torch.load(str(pth), map_location="cpu")
    return state.get("model", state)

ckpt = {
    "stage1": load_state_dict(S1_PTH),
    "stage2": load_state_dict(S2_PTH),
    "classes1": classes1,
    "classes2": classes2,
    "tau1": tau1,
    "tau2": tau2,
}

OUT = MODELS_BASE / "maize_two_stage.pth"
torch.save(ckpt, OUT)
print("🎉 Bundle guardado en:", OUT.resolve())
