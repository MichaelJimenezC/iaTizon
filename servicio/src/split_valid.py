import os, random, shutil
from pathlib import Path

root = Path(r"C:\dev\dataset")
train_dir = root / "train"
valid_dir = root / "valid"

val_ratio = 0.2

classes = [c.name for c in train_dir.iterdir() if c.is_dir() and c.name != "tizon_foliar"]

for cls in classes:
    src_dir = train_dir / cls
    dst_dir = valid_dir / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    imgs = [f for f in src_dir.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png"]]
    random.shuffle(imgs)

    n_val = int(len(imgs) * val_ratio)
    val_imgs = imgs[:n_val]

    for img in val_imgs:
        shutil.copy2(img, dst_dir / img.name)

    print(f"{cls}: {n_val} imágenes movidas a valid/")
