import os, random, shutil
from pathlib import Path

root = Path(r"C:\dev\dataset")
train_dir = root / "train"
test_dir = root / "test"

test_ratio = 0.1  

test_dir.mkdir(parents=True, exist_ok=True)

classes = [c.name for c in train_dir.iterdir() if c.is_dir() and c.name != "tizon_foliar"]

for cls in classes:
    src_dir = train_dir / cls
    dst_dir = test_dir / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    imgs = [f for f in src_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(imgs)

    n_test = int(len(imgs) * test_ratio)
    test_imgs = imgs[:n_test]

    for img in test_imgs:
        shutil.move(str(img), dst_dir / img.name)  

    print(f"{cls}: {n_test} imágenes movidas a test/")

print("✅ División de test completada (tizon_foliar no modificado).")
