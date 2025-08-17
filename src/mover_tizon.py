import os
import random
import shutil

# RUTA BASE DEL DATASET
BASE_DIR = r"C:\dev\dataset"

# CUÁNTAS IMÁGENES MOVER
MOVER_A_VALID = 150
MOVER_A_TEST = 100

# CLASE A MOVER
CLASE = "tizon_foliar"

# RUTAS
train_dir = os.path.join(BASE_DIR, "train", CLASE)
valid_dir = os.path.join(BASE_DIR, "valid", CLASE)
test_dir = os.path.join(BASE_DIR, "test", CLASE)

# Crear carpetas destino si no existen
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Listar imágenes en train
imagenes = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
random.shuffle(imagenes)

# Mover a VALID
for img in imagenes[:MOVER_A_VALID]:
    shutil.move(os.path.join(train_dir, img), os.path.join(valid_dir, img))

# Mover a TEST
for img in imagenes[MOVER_A_VALID:MOVER_A_VALID + MOVER_A_TEST]:
    shutil.move(os.path.join(train_dir, img), os.path.join(test_dir, img))

print(f"Movidas {MOVER_A_VALID} imágenes a VALID y {MOVER_A_TEST} a TEST para la clase '{CLASE}'")
