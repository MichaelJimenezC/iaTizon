import os
import shutil
import random

# Paths
SRC_DIR = r"C:\Users\micha\Downloads\archive (5)\data"
DST_DIR = r"C:\dev\dataset_maize_only"

# Map source → target class
CLASS_MAP = {
    "Healthy": "healthy",
    "Blight": "tizon_foliar"
}

# Split ratios
TRAIN_RATIO = 0.8
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# Crear estructura de carpetas si no existe
for split in ["train", "valid", "test"]:
    for target_class in CLASS_MAP.values():
        path = os.path.join(DST_DIR, split, target_class)
        os.makedirs(path, exist_ok=True)

# Procesar cada clase mapeada
for src_class, target_class in CLASS_MAP.items():
    src_path = os.path.join(SRC_DIR, src_class)
    if not os.path.exists(src_path):
        print(f"⚠️ Carpeta {src_path} no existe, la salto...")
        continue

    files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_valid = int(n_total * VALID_RATIO)
    n_test  = n_total - n_train - n_valid

    splits = {
        "train": files[:n_train],
        "valid": files[n_train:n_train+n_valid],
        "test":  files[n_train+n_valid:]
    }

    for split, split_files in splits.items():
        for f in split_files:
            src_file = os.path.join(src_path, f)
            dst_file = os.path.join(DST_DIR, split, target_class, f)
            shutil.copy2(src_file, dst_file)

    print(f"✅ {src_class} → {target_class}: {n_total} imágenes copiadas")

print("🎉 Dataset actualizado en", DST_DIR)
