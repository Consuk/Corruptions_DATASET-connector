import os
from PIL import Image
import numpy as np
from endoscopycorruptions import corrupt, get_corruption_names
from tqdm import tqdm

# Configuración de rutas
TEST_LIST = "/workspace/datasets/hamlyn/splits/test_files2.txt"
INPUT_ROOT = "/workspace/datasets/hamlyn/Hamlyn"
OUTPUT_ROOT = "/workspace/datasets/hamlyn/hamlyn_corruptions_test"

corruption_types = get_corruption_names()
severities = [1, 2, 3, 4, 5]

# Leer lista de test
with open(TEST_LIST, "r") as f:
    test_lines = [line.strip().split() for line in f if line.strip()]

for parts in tqdm(test_lines, desc="Corrompiendo test set"):
    rel_path, img_idx = parts[0], parts[1]
    side = parts[2] if len(parts) >= 3 else None

    # normaliza si viene con .jpg o sin ceros
    img_idx = os.path.splitext(img_idx)[0]
    if img_idx.isdigit() and len(img_idx) < 10:
        img_idx = img_idx.zfill(10)

    # Caso 1: el split ya incluye image01/image02 en rel_path
    if rel_path.endswith(("image01", "image02")):
        img_path = os.path.join(INPUT_ROOT, rel_path, f"{img_idx}.jpg")
        out_rel = rel_path
    else:
        # Caso 2: rel_path es la secuencia y la 3ra columna dice l/r
        if side is None:
            raise ValueError("Tu split debe tener 3ra columna l/r o incluir image01/image02 en rel_path")
        cam = "image01" if side.lower().startswith("l") else "image02"
        img_path = os.path.join(INPUT_ROOT, rel_path, cam, f"{img_idx}.jpg")
        out_rel = os.path.join(rel_path, cam)


    if not os.path.exists(img_path):
        print(f"[WARNING] No encontrada: {img_path}")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img)
    except Exception as e:
        print(f"[ERROR] Fallo al cargar {img_path}: {e}")
        continue

    for corr in corruption_types:
        for sev in severities:
            try:
                img_corr = corrupt(img_np, corruption_name=corr, severity=sev)
                out_dir = os.path.join(OUTPUT_ROOT, corr, f"severity_{sev}", out_rel)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{img_idx}.jpg")
                Image.fromarray(img_corr).save(out_path)
            except Exception as e:
                print(f"[ERROR] {corr} s{sev} - {rel_path}/{img_idx}.jpg: {e}")
