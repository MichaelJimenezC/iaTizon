import argparse, os, shutil
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def link_or_copy(src: Path, dst: Path, prefer_hardlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)            
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def build_stage1(src: Path, dst_bin: Path, split: str, neg_extras: Path|None, prefer_hardlink: bool):
    """
    Etapa 1: maize vs not_maize
    maize = healthy + tizon_foliar
    not_maize = not_maize (+ neg_extras opcional)
    """
    mapping = {
        "healthy":      ("maize", True),
        "tizon_foliar": ("maize", True),
        "not_maize":    ("not_maize", True),
    }
    for cls, (new_cls, use_src) in mapping.items():
        src_dir = src / split / cls
        if use_src and src_dir.exists():
            for f in src_dir.rglob("*"):
                if is_img(f):
                    link_or_copy(f, dst_bin / split / new_cls / f.name, prefer_hardlink)

    if neg_extras and neg_extras.exists():
        for f in neg_extras.rglob("*"):
            if is_img(f):
                link_or_copy(f, dst_bin / split / "not_maize" / f.name, prefer_hardlink)

def build_stage2(src: Path, dst_maize: Path, split: str, prefer_hardlink: bool):
    """
    Etapa 2: healthy vs tizon_foliar (solo maíz)
    """
    for cls in ["healthy", "tizon_foliar"]:
        src_dir = src / split / cls
        if src_dir.exists():
            for f in src_dir.rglob("*"):
                if is_img(f):
                    link_or_copy(f, dst_maize / split / cls / f.name, prefer_hardlink)

def main():
    ap = argparse.ArgumentParser(description="Crear datasets derivados para pipeline en 2 etapas.")
    ap.add_argument("--src", required=True, help="Dataset original con train/valid/test y healthy/not_maize/tizon_foliar")
    ap.add_argument("--dst_bin", default="C:\\dev\\dataset_bin", help="Salida etapa 1 (maize vs not_maize)")
    ap.add_argument("--dst_maize", default="C:\\dev\\dataset_maize_only", help="Salida etapa 2 (healthy vs tizon_foliar)")
    ap.add_argument("--neg_extras", default="", help="Carpeta opcional con negativos duros para not_maize")
    ap.add_argument("--copy", action="store_true", help="Forzar copia (por defecto intenta hardlink)")
    args = ap.parse_args()

    src = Path(args.src)
    dst_bin = Path(args.dst_bin)
    dst_maize = Path(args.dst_maize)
    neg_extras = Path(args.neg_extras) if args.neg_extras else None
    prefer_hardlink = not args.copy

    if not (src / "train").exists():
        raise SystemExit("❌ No encuentro carpetas train/valid/test en --src")

    for split in ["train","valid","test"]:
        build_stage1(src, dst_bin, split, neg_extras, prefer_hardlink)
        build_stage2(src, dst_maize, split, prefer_hardlink)

    print("✅ Listo.")
    print(f" - Etapa 1 (binario): {dst_bin}")
    print(f" - Etapa 2 (solo maíz): {dst_maize}")
    if neg_extras and neg_extras.exists():
        print(f" - Negativos duros añadidos desde: {neg_extras}")

if __name__ == "__main__":
    main()
