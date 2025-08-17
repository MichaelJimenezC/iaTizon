import os
import random
import shutil

# Ruta al COCO 2017 train (descomprimido)
COCO_TRAIN_DIR = r"C:\Users\micha\Downloads\train2017\train2017"

# Tu dataset destino
NEGATIVOS_DIR = r"C:\dev\dataset\negativos"

# Cuántas imágenes quieres muestrear
N_IMGS = 5000

# Crear carpeta si no existe
os.makedirs(NEGATIVOS_DIR, exist_ok=True)

# Listar imágenes disponibles
all_images = [f for f in os.listdir(COCO_TRAIN_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Selección aleatoria
sampled_images = random.sample(all_images, min(N_IMGS, len(all_images)))

print(f"Copiando {len(sampled_images)} imágenes a {NEGATIVOS_DIR}...")

# Copiar archivos
for img in sampled_images:
    src = os.path.join(COCO_TRAIN_DIR, img)
    dst = os.path.join(NEGATIVOS_DIR, img)
    shutil.copy2(src, dst)

print("✅ Listo: imágenes copiadas.")
