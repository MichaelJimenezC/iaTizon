import os
import random
import shutil

# Carpetas origen y destino
SRC_DIR = r"C:\Users\micha\Downloads\test"   # Sorgo
DST_DIR = r"C:\dev\parecidas\sorgo"          # Carpeta destino (subcarpeta sorgo)

# Número de imágenes a seleccionar
N_IMGS = 500

# Crear carpeta destino si no existe
os.makedirs(DST_DIR, exist_ok=True)

# Listar imágenes disponibles
all_images = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Selección aleatoria
sampled_images = random.sample(all_images, min(N_IMGS, len(all_images)))

print(f"Copiando {len(sampled_images)} imágenes a {DST_DIR}...")

# Copiar archivos
for img in sampled_images:
    src = os.path.join(SRC_DIR, img)
    dst = os.path.join(DST_DIR, img)
    shutil.copy2(src, dst)

print("✅ Listo: imágenes copiadas.")
