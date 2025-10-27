import os
from PIL import Image
import numpy as np
from endoscopycorruptions import corrupt, get_corruption_names
from tqdm import tqdm

# Configuración de rutas
TEST_LIST = "/workspace/AF-SfMLearner/splits/endovis/test_files.txt"
INPUT_ROOT = "/workspace/AF-SfMLearner/endovis_data"
OUTPUT_ROOT = "/workspace/endovis_corruptions_test"

corruption_types = get_corruption_names()
severities = [1, 2, 3, 4, 5]

# Leer lista de test
with open(TEST_LIST, "r") as f:
    test_lines = [line.strip().split()[0:2] for line in f if line.strip()]

# Procesar cada imagen del test set
for rel_path, img_idx in tqdm(test_lines, desc="Corrompiendo test set"):
    img_path = os.path.join(INPUT_ROOT, rel_path, "data", f"{img_idx}.jpg")

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
                out_dir = os.path.join(OUTPUT_ROOT, corr, f"severity_{sev}", rel_path, "data")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{img_idx}.jpg")
                Image.fromarray(img_corr).save(out_path)
            except Exception as e:
                print(f"[ERROR] {corr} s{sev} - {rel_path}/{img_idx}.jpg: {e}")
