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
        if p.suffix.lower() in exts:
            yield str(p)

@torch.no_grad()
def predict_one(model, device, tfm, path, classes, tau=None):
    """
    Si tau no es None y las clases contienen 'maize' y 'not_maize',
    aplica regla: si P(maize) < tau => pred = 'not_maize'.
    """
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device, non_blocking=True)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_name = classes[pred_idx]

    # Reglas anti-falsos positivos para stage1 (opcional)
    if tau is not None and "maize" in classes and "not_maize" in classes and len(classes) == 2:
        i_maize = classes.index("maize")
        i_not   = classes.index("not_maize")
        p_maize = float(probs[i_maize])
        if p_maize < tau:
            pred_idx = i_not
            pred_name = "not_maize"

    return pred_name, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Raíz del dataset usado para entrenar (para leer el orden de clases)")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pth")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--img", type=str, default=None, help="Ruta a una sola imagen")
    ap.add_argument("--dir", type=str, default=None, help="Carpeta (se busca recursivo)")
    ap.add_argument("--out_csv", type=str, default="preds.csv")
    ap.add_argument("--tau", type=float, default=None, help="(Solo stage1 binario) si P(maize) < tau => 'not_maize'")
    args = ap.parse_args()

    # Dispositivo y preproc
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tfm = load_tf(args.img_size)

    # Orden de clases (del dataset)
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"))
    classes = train_ds.classes  # p.ej. ['maize','not_maize'] o ['healthy','tizon_foliar'] o 3 clases

    # Modelo
    model = ResNet18(num_classes=len(classes)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # Recolectar paths
    paths = []
    if args.img:
        paths = [args.img]
    elif args.dir:
        paths = sorted(iter_images(args.dir))
    else:
        print("Provee --img o --dir"); return

    # CSV dinámico según número de clases
    header = ["path","pred"] + [f"prob_{c}" for c in classes]
    rows = [header]

    for p in paths:
        pred_name, probs = predict_one(model, device, tfm, p, classes, tau=args.tau)
        rows.append([p, pred_name] + [float(probs[i]) for i in range(len(classes))])

    # Guardar
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"✅ ok -> {args.out_csv}")
    print(f"Clases: {classes}")
    if args.tau is not None:
        print(f"Umbral tau aplicado (solo si binario maize/not_maize): {args.tau}")

if __name__ == "__main__":
    main()
