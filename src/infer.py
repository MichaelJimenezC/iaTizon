import argparse, os, csv, json, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets
import torch.nn.functional as F
from .data import MEAN, STD
from .model import ResNet18

def load_tf(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def iter_images(folder):
    exts={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield str(p)

@torch.no_grad()
def predict_one(model, device, tfm, path, classes, tau=None, tau_pos_class=None):
    """
    - Si tau y tau_pos_class están definidos y es binario:
        si P(pos_class) >= tau => pred = pos_class; si no => la otra clase.
    - Si no, usa argmax normal.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return "__error__", [0.0]*len(classes)

    x = tfm(img).unsqueeze(0).to(device, non_blocking=True)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    if tau is not None and tau_pos_class is not None and len(classes)==2 and (tau_pos_class in classes):
        pos_idx = classes.index(tau_pos_class)
        p_pos   = float(probs[pos_idx])
        pred_idx = pos_idx if p_pos >= tau else 1 - pos_idx
        return classes[pred_idx], probs

    if tau is not None and "maize" in classes and "not_maize" in classes and len(classes) == 2:
        i_maize = classes.index("maize")
        i_not   = classes.index("not_maize")
        p_maize = float(probs[i_maize])
        pred_idx = i_not if p_maize < tau else i_maize
        return classes[pred_idx], probs

    pred_idx = int(probs.argmax())
    return classes[pred_idx], probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Raíz del dataset usado para entrenar (para leer el orden de clases)")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pth")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--img", type=str, default=None, help="Ruta a una sola imagen")
    ap.add_argument("--dir", type=str, default=None, help="Carpeta (se busca recursivo)")
    ap.add_argument("--out_csv", type=str, default="preds.csv")
    ap.add_argument("--tau", type=float, default=None, help="Umbral para binario (opcional)")
    ap.add_argument("--tau_pos_class", type=str, default=None, help="Nombre de la clase positiva para --tau (ej. 'maize' o 'tizon_foliar')")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tfm = load_tf(args.img_size)

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"))
    classes = train_ds.classes 

    model = ResNet18(num_classes=len(classes)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    if args.img:
        paths = [args.img]
    elif args.dir:
        paths = sorted(iter_images(args.dir))
    else:
        print("Provee --img o --dir"); return

    if not paths:
        print(f"⚠️  No encontré imágenes en: {args.dir}")
        return

    header = ["path","pred"] + [f"prob_{c}" for c in classes]
    rows = [header]

    errores = 0
    for p in paths:
        pred_name, probs = predict_one(model, device, tfm, p, classes, tau=args.tau, tau_pos_class=args.tau_pos_class)
        if pred_name == "__error__":
            errores += 1
            continue
        rows.append([p, pred_name] + [float(probs[i]) for i in range(len(classes))])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"✅ ok -> {args.out_csv}")
    print(f"Clases: {classes}")
    print(f"Imgs procesadas: {len(paths)-errores}  |  dañadas: {errores}")
    if args.tau is not None:
        if args.tau_pos_class:
            print(f"Umbral tau aplicado: {args.tau} (pos='{args.tau_pos_class}')")
        else:
            print(f"Umbral tau aplicado (modo legacy maize/not_maize): {args.tau}")

if __name__ == "__main__":
    main()
