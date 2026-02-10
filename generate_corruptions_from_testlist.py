import os
from PIL import Image
import numpy as np
from endoscopycorruptions import corrupt, get_corruption_names
from tqdm import tqdm

# =========================
# Configuración de rutas
# =========================
TEST_LIST   = "/workspace/datasets/hamlyn/splits/test_files2.txt"
INPUT_ROOT  = "/workspace/datasets/hamlyn/Hamlyn"
OUTPUT_ROOT = "/workspace/datasets/hamlyn/hamlyn_corruptions_test"

corruption_types = get_corruption_names()
severities = [1, 2, 3, 4, 5]


def normalize_img_idx(img_idx: str) -> str:
    """Normaliza el índice: quita extensión y rellena con ceros a 10 dígitos si es numérico."""
    img_idx = os.path.splitext(img_idx)[0]
    if img_idx.isdigit() and len(img_idx) < 10:
        img_idx = img_idx.zfill(10)
    return img_idx


def resolve_hamlyn_image_path(input_root: str, rel_path: str, cam: str, img_idx: str):
    """
    Resuelve rutas Hamlyn soportando ambas variantes:

    1) input_root/rel_path/cam/img.jpg
       ej: Hamlyn/rectified05/image01/0000.jpg   (NO es tu caso, pero se soporta)

    2) input_root/rel_path/rel_path/cam/img.jpg
       ej: Hamlyn/rectified05/rectified05/image01/0000.jpg  (TU CASO REAL)
    """
    cand1 = os.path.join(input_root, rel_path, cam, f"{img_idx}.jpg")
    cand2 = os.path.join(input_root, rel_path, rel_path, cam, f"{img_idx}.jpg")

    if os.path.isfile(cand1):
        # out_rel mantiene la estructura relativa para output_root
        return cand1, os.path.join(rel_path, cam)

    if os.path.isfile(cand2):
        return cand2, os.path.join(rel_path, rel_path, cam)

    return None, None


# =========================
# Leer lista de test
# =========================
with open(TEST_LIST, "r") as f:
    test_lines = [line.strip().split() for line in f if line.strip()]

# =========================
# Procesar
# =========================
for parts in tqdm(test_lines, desc="Corrompiendo test set"):
    rel_path, img_idx = parts[0], parts[1]
    side = parts[2] if len(parts) >= 3 else None

    img_idx = normalize_img_idx(img_idx)

    # Caso 1: el split ya incluye image01/image02 en rel_path
    if rel_path.endswith(("image01", "image02")):
        # Aquí también soportamos doble-rectified si rel_path trae solo una parte (raro pero posible)
        # Si rel_path ya es ".../rectifiedXX/rectifiedXX/image01" entonces cand1 funciona directo.
        cand1 = os.path.join(INPUT_ROOT, rel_path, f"{img_idx}.jpg")
        cand2 = None

        # Si rel_path es tipo "rectified05/image01" y existe la variante doble
        # (input_root/rectified05/rectified05/image01/img.jpg), intentamos corregirlo:
        if not os.path.isfile(cand1):
            pieces = rel_path.split("/")
            if len(pieces) >= 2 and pieces[0].startswith("rectified") and pieces[1] in ("image01", "image02"):
                rect = pieces[0]
                cam = pieces[1]
                cand2 = os.path.join(INPUT_ROOT, rect, rect, cam, f"{img_idx}.jpg")

        if os.path.isfile(cand1):
            img_path = cand1
            out_rel = rel_path
        elif cand2 and os.path.isfile(cand2):
            img_path = cand2
            # output mantiene la ruta correcta (rectifiedXX/rectifiedXX/image01)
            out_rel = os.path.join(pieces[0], pieces[0], pieces[1])
        else:
            print(f"[WARNING] No encontrada: {cand1}")
            if cand2:
                print(f"[WARNING] No encontrada: {cand2}")
            continue

    else:
        # Caso 2: rel_path es la secuencia y la 3ra columna dice l/r
        if side is None:
            raise ValueError(
                "Tu split debe tener 3ra columna l/r o incluir image01/image02 en rel_path"
            )

        cam = "image01" if side.lower().startswith("l") else "image02"

        img_path, out_rel = resolve_hamlyn_image_path(INPUT_ROOT, rel_path, cam, img_idx)

        if img_path is None:
            # Log detallado
            cand1 = os.path.join(INPUT_ROOT, rel_path, cam, f"{img_idx}.jpg")
            cand2 = os.path.join(INPUT_ROOT, rel_path, rel_path, cam, f"{img_idx}.jpg")
            print(f"[WARNING] No encontrada: {cand1}")
            print(f"[WARNING] No encontrada: {cand2}")
            continue

    # Cargar imagen
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img)
    except Exception as e:
        print(f"[ERROR] Fallo al cargar {img_path}: {e}")
        continue

    # Corromper + guardar
    for corr in corruption_types:
        for sev in severities:
            try:
                img_corr = corrupt(img_np, corruption_name=corr, severity=sev)

                out_dir = os.path.join(
                    OUTPUT_ROOT,
                    corr,
                    f"severity_{sev}",
                    out_rel
                )
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f"{img_idx}.jpg")
                Image.fromarray(img_corr).save(out_path)

            except Exception as e:
                print(f"[ERROR] {corr} s{sev} - {out_rel}/{img_idx}.jpg: {e}")
