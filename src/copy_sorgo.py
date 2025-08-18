import os
import random
import shutil

SRC_DIR = r"C:\Users\micha\Downloads\test"   
DST_DIR = r"C:\dev\parecidas\sorgo"          

N_IMGS = 500

os.makedirs(DST_DIR, exist_ok=True)

all_images = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

sampled_images = random.sample(all_images, min(N_IMGS, len(all_images)))

print(f"Copiando {len(sampled_images)} imágenes a {DST_DIR}...")

for img in sampled_images:
    src = os.path.join(SRC_DIR, img)
    dst = os.path.join(DST_DIR, img)
    shutil.copy2(src, dst)

print("✅ Listo: imágenes copiadas.")
