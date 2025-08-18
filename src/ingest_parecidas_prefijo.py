import os, random, shutil, argparse
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

def main():
    ap = argparse.ArgumentParser("Ingerir 'parecidas' -> dataset_bin/*/not_maize (80/10/10) con prefijo por subcarpeta")
    ap.add_argument("--src", default=r"C:\dev\parecidas", help="Raíz con subcarpetas (arroz, sorgo, etc.)")
    ap.add_argument("--dst_bin", default=r"C:\dev\dataset_bin", help="Raíz del dataset binario")
    ap.add_argument("--ratios", type=str, default="0.8,0.1,0.1", help="train,valid,test")
    ap.add_argument("--copy", action="store_true", help="Forzar copia (no hardlinks)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true", help="No sobrescribir si ya existe destino")
    args = ap.parse_args()

    random.seed(args.seed)
    r_train, r_valid, r_test = map(float, args.ratios.split(","))
    src_root = Path(args.src)
    dst_root = Path(args.dst_bin)
    prefer_hardlink = not args.copy

    for split in ["train", "valid", "test"]:
        (dst_root / split / "not_maize").mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"⚠️ No hay subcarpetas en {src_root}")
        return

    total = 0
    for sd in sorted(subdirs):
        label = sd.name.lower()
        imgs = collect_images(sd)
        if not imgs:
            print(f"[skip] {label}: 0 imágenes")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_tr = int(n * r_train)
        n_va = int(n * r_valid)
        splits = {
            "train": imgs[:n_tr],
            "valid": imgs[n_tr:n_tr+n_va],
            "test" : imgs[n_tr+n_va:]
        }

        for split, lst in splits.items():
            for i, f in enumerate(lst):
                dst_name = f"{label}__{f.name}"
                dst = dst_root / split / "not_maize" / dst_name
                if args.skip_existing and dst.exists():
                    continue
                link_or_copy(f, dst, prefer_hardlink)

        total += n
        print(f"[OK] {label:10s}: total={n}  -> train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

    print(f"\n✅ Listo. Ingeridas {total} imágenes a not_maize.")

if __name__ == "__main__":
    main()
