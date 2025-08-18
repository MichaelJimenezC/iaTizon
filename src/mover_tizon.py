import os
import random
import shutil

BASE_DIR = r"C:\dev\dataset"

MOVER_A_VALID = 150
MOVER_A_TEST = 100

CLASE = "tizon_foliar"

train_dir = os.path.join(BASE_DIR, "train", CLASE)
valid_dir = os.path.join(BASE_DIR, "valid", CLASE)
test_dir = os.path.join(BASE_DIR, "test", CLASE)

os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

imagenes = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
random.shuffle(imagenes)

for img in imagenes[:MOVER_A_VALID]:
    shutil.move(os.path.join(train_dir, img), os.path.join(valid_dir, img))

for img in imagenes[MOVER_A_VALID:MOVER_A_VALID + MOVER_A_TEST]:
    shutil.move(os.path.join(train_dir, img), os.path.join(test_dir, img))

print(f"Movidas {MOVER_A_VALID} imágenes a VALID y {MOVER_A_TEST} a TEST para la clase '{CLASE}'")
