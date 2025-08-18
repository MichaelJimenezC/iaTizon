import os
import random
import shutil

COCO_TRAIN_DIR = r"C:\Users\micha\Downloads\train2017\train2017"

PRUEBAS_DIR = r"C:\dev\pruebas"

N_IMGS = 200   

os.makedirs(PRUEBAS_DIR, exist_ok=True)

all_images = [f for f in os.listdir(COCO_TRAIN_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

sampled_images = random.sample(all_images, min(N_IMGS, len(all_images)))

print(f"Copiando {len(sampled_images)} imágenes a {PRUEBAS_DIR}...")

for img in sampled_images:
    src = os.path.join(COCO_TRAIN_DIR, img)
    dst = os.path.join(PRUEBAS_DIR, img)
    shutil.copy2(src, dst)

print("✅ Listo: carpeta de pruebas poblada.")
