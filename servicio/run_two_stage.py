# C:\dev\servicio\run_two_stage.py
import os, json
from io import BytesIO
import argparse, torch, torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from src.model import ResNet18
from src.data import MEAN, STD

BASE = r"/Users/user1/Documents/GitHub/iaTizon/servicio/models"
S1 = os.path.join(BASE, "stage1")
S2 = os.path.join(BASE, "stage2")

def load_meta(dir_):
    classes = json.load(open(os.path.join(dir_, "classes.json"), "r", encoding="utf-8"))
    tau_path = os.path.join(dir_, "tau.txt")
    tau = float(open(tau_path).read().strip()) if os.path.exists(tau_path) else None
    return classes, tau

CLASSES1, TAU1 = load_meta(S1)   # ["maize","not_maize"], tau≈0.272
CLASSES2, TAU2 = load_meta(S2)   # ["healthy","tizon_foliar"], tau≈0.395 (o None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TFM = transforms.Compose([
    transforms.Resize(int(320*1.14)),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_model(pth, ncls):
    m = ResNet18(num_classes=ncls).to(DEVICE).eval()
    state = torch.load(pth, map_location=DEVICE)
    state = state.get("model", state)
    m.load_state_dict(state, strict=False)
    return m

M1 = load_model(os.path.join(S1, "stage1_maize_not_v3.pth"), len(CLASSES1))
M2 = load_model(os.path.join(S2, "stage2_maize_tizon.pth"), len(CLASSES2))


def probs(model, img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = TFM(img).unsqueeze(0).to(DEVICE, non_blocking=True)
    with torch.no_grad():
        return F.softmax(model(x),1)[0].cpu().tolist()

def apply_tau(probs, classes, tau, pos_class):
    if tau is None or len(classes)!=2 or pos_class not in classes:
        return classes[int(torch.tensor(probs).argmax())]
    i = classes.index(pos_class)
    return pos_class if probs[i] >= tau else classes[1-i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    args = ap.parse_args()

    img = Image.open(args.img)

    p1 = probs(M1, img)
    pred1 = apply_tau(p1, CLASSES1, TAU1, pos_class="maize")
    if pred1 != "maize":
        print({"final":"not_maize","stage1":dict(zip(CLASSES1, map(float,p1)))})
        return

    p2 = probs(M2, img)
    pred2 = apply_tau(p2, CLASSES2, TAU2, pos_class="tizon_foliar")
    print({"final":pred2,
           "stage1":dict(zip(CLASSES1, map(float,p1))),
           "stage2":dict(zip(CLASSES2, map(float,p2)))})

if __name__ == "__main__":
    main()
