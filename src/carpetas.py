import os

DATASET_DIR = r"C:\dev\dataset"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def contar_imagenes_en_carpeta(path):
    """Cuenta imágenes con extensiones válidas en un directorio."""
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                count += 1
    return count

def main():
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_path):
            continue

        print(f"\n📂 {split.upper()}")
        total_split = 0
        for clase in sorted(os.listdir(split_path)):
            clase_path = os.path.join(split_path, clase)
            if os.path.isdir(clase_path):
                n_imgs = contar_imagenes_en_carpeta(clase_path)
                total_split += n_imgs
                print(f"  {clase:<20} -> {n_imgs} imágenes")
        print(f"  Total {split}: {total_split} imágenes")

if __name__ == "__main__":
    main()
