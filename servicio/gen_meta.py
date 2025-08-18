import argparse, json, os
from torchvision import datasets

def save_meta(out_dir, classes_from_train, tau):
    os.makedirs(out_dir, exist_ok=True)
    classes = datasets.ImageFolder(classes_from_train).classes
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)
    if tau is not None:
        with open(os.path.join(out_dir, "tau.txt"), "w", encoding="utf-8") as f:
            f.write(str(tau))
    print(f"[OK] {out_dir} -> classes={classes}  tau={tau}")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["stage1","stage2"], required=True)
ap.add_argument("--train_dir", required=True)         
ap.add_argument("--tau", type=float, default=None)     
args = ap.parse_args()

base = r"C:\dev\servicio\models"
out = os.path.join(base, args.stage)
save_meta(out, args.train_dir, args.tau)
