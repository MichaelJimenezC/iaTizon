import argparse, os, shutil, random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def link_or_copy(src: Path, dst: Path, prefer_hardlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)          
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def collect_images(root: Path):
    return [p for p in root.rglob("*") if is_img(p)]

def copy_split_maize(src_root: Path, dst_bin_root: Path, dst_maize_root: Path,
                     split: str, prefer_hardlink: bool):
    # ---- Etapa 1 (binario): maize = healthy + tizon_foliar
    for cls in ["healthy", "tizon_foliar"]:
        cdir = src_root / split / cls
        if not cdir.exists(): continue
        for f in cdir.rglob("*"):
            if is_img(f):
                link_or_copy(f, dst_bin_root / split / "maize" / f.name, prefer_hardlink)

    nm_dir = src_root / split / "not_maize"
    if nm_dir.exists():
        for f in nm_dir.rglob("*"):
            if is_img(f):
                link_or_copy(f, dst_bin_root / split / "not_maize" / f.name, prefer_hardlink)

    # ---- Etapa 2 (solo maíz): healthy y tizon_foliar
    for cls in ["healthy", "tizon_foliar"]:
        cdir = src_root / split / cls
        if not cdir.exists(): continue
        for f in cdir.rglob("*"):
            if is_img(f):
                link_or_copy(f, dst_maize_root / split / cls / f.name, prefer_hardlink)

def split_and_place_negatives(neg_root: Path, dst_bin_root: Path,
                              ratios=(0.8, 0.1, 0.1), seed=42,
                              prefer_hardlink=True):
    """Reparte dataset/negativos en train/valid/test -> not_maize."""
    if not neg_root.exists():
        print(f"[WARN] No existe {neg_root}; omito reparto de negativos extra.")
        return

    imgs = collect_images(neg_root)
    if not imgs:
        print(f"[WARN] {neg_root} está vacío.")
        return

    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_imgs = imgs[:n_train]
    valid_imgs = imgs[n_train:n_train+n_valid]
    test_imgs  = imgs[n_train+n_valid:]

    for f in train_imgs:
        link_or_copy(f, dst_bin_root / "train" / "not_maize" / f.name, prefer_hardlink)
    for f in valid_imgs:
        link_or_copy(f, dst_bin_root / "valid" / "not_maize" / f.name, prefer_hardlink)
    for f in test_imgs:
        link_or_copy(f, dst_bin_root / "test"  / "not_maize" / f.name, prefer_hardlink)

    print(f"[OK] Negativos extra repartidos: train={len(train_imgs)}, valid={len(valid_imgs)}, test={len(test_imgs)}")

def count_summary(root: Path, title: str):
    print(f"\n== {title} ==")
    for split in ["train", "valid", "test"]:
        base = root / split
        if not base.exists():
            print(f"[{split}] (no existe)")
            continue
        print(f"[{split}]")
        for cls in sorted([p for p in base.iterdir() if p.is_dir()]):
            n = sum(1 for p in cls.rglob("*") if is_img(p))
            print(f"  {cls.name:15s}: {n}")

def main():
    ap = argparse.ArgumentParser(description="Construye datasets para pipeline 2 etapas (hardlinks).")
    ap.add_argument("--src", default=r"C:\dev\dataset", help="dataset original (3 clases + negativos)")
    ap.add_argument("--dst_bin", default=r"C:\dev\dataset_bin", help="salida etapa1 (maize vs not_maize)")
    ap.add_argument("--dst_maize", default=r"C:\dev\dataset_maize_only", help="salida etapa2 (healthy vs tizon_foliar)")
    ap.add_argument("--neg_dir", default=r"C:\dev\dataset\negativos", help="carpeta con negativos extra (sin split)")
    ap.add_argument("--ratios", type=str, default="0.8,0.1,0.1", help="proporción train,valid,test para negativos (suma 1.0)")
    ap.add_argument("--copy", action="store_true", help="forzar copia en vez de hardlink")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst_bin = Path(args.dst_bin)
    dst_maize = Path(args.dst_maize)
    neg_dir = Path(args.neg_dir)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    prefer_hardlink = not args.copy

    for split in ["train","valid","test"]:
        (dst_bin / split / "maize").mkdir(parents=True, exist_ok=True)
        (dst_bin / split / "not_maize").mkdir(parents=True, exist_ok=True)
        (dst_maize / split / "healthy").mkdir(parents=True, exist_ok=True)
        (dst_maize / split / "tizon_foliar").mkdir(parents=True, exist_ok=True)

    for split in ["train","valid","test"]:
        copy_split_maize(src, dst_bin, dst_maize, split, prefer_hardlink)

    split_and_place_negatives(neg_dir, dst_bin, ratios=ratios, seed=args.seed, prefer_hardlink=prefer_hardlink)

    count_summary(dst_bin, "dataset_bin (maize vs not_maize)")
    count_summary(dst_maize, "dataset_maize_only (healthy vs tizon_foliar)")
    print("\n✅ Listo.")

if __name__ == "__main__":
    main()
