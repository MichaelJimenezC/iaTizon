import argparse, os, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def is_img(p: Path): 
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def link_or_copy(src: Path, dst: Path, prefer_hardlink=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)  
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Añadir imágenes extra de MAIZE a dataset_bin con split 80/10/10.")
    ap.add_argument("--src_maize", required=True, help="Carpeta con las 7000 imágenes de maíz")
    ap.add_argument("--dst_bin", default=r"C:\dev\dataset_bin", help="Raíz del dataset binario (maize vs not_maize)")
    ap.add_argument("--ratios", type=str, default="0.8,0.1,0.1", help="train,valid,test (suma 1.0)")
    ap.add_argument("--copy", action="store_true", help="Forzar copia en vez de hardlink")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src_maize)
    dst = Path(args.dst_bin)
    r_train, r_valid, r_test = map(float, args.ratios.split(","))
    prefer_hardlink = not args.copy

    imgs = [p for p in src.rglob("*") if is_img(p)]
    if not imgs:
        raise SystemExit(f"❌ No se encontraron imágenes en {src}")

    random.Random(args.seed).shuffle(imgs)
    n = len(imgs)
    n_train = int(n * r_train)
    n_valid = int(n * r_valid)
    train_imgs = imgs[:n_train]
    valid_imgs = imgs[n_train:n_train+n_valid]
    test_imgs  = imgs[n_train+n_valid:]

    for p in train_imgs:
        link_or_copy(p, dst / "train" / "maize" / p.name, prefer_hardlink)
    for p in valid_imgs:
        link_or_copy(p, dst / "valid" / "maize" / p.name, prefer_hardlink)
    for p in test_imgs:
        link_or_copy(p, dst / "test"  / "maize" / p.name, prefer_hardlink)

    print(f"✅ Añadido a {dst}: train={len(train_imgs)}, valid={len(valid_imgs)}, test={len(test_imgs)}")

if __name__ == "__main__":
    main()
