import os
from pathlib import Path

# Ruta al directorio Hamlyn (ajusta esto si lo tienes en otra ubicación)
BASE_PATH = Path("/workspace/datasets/hamlyn/Hamlyn")

# Ojo izquierdo (image01) y etiqueta correspondiente
CAMERA_DIR = "image01"
CAMERA_LABEL = "l"

# División de secuencias por conjunto
SEQUENCES = {
    "train": ["rectified01", "rectified05", "rectified06", "rectified09", "rectified11", "rectified14", "rectified27"],
    "val": ["rectified12", "rectified15"],
    "test": ["rectified04", "rectified08"]
}

# Carpeta donde guardar los .txt resultantes
OUTPUT_DIR = BASE_PATH.parent  # mismo nivel que Hamlyn/

def generate_split_txts():
    for split, folders in SEQUENCES.items():
        lines = []
        for folder in folders:
            image_dir = BASE_PATH / folder / folder / CAMERA_DIR
            if not image_dir.exists():
                print(f"[WARN] No se encontró: {image_dir}")
                continue

            # Leer todos los archivos .png ordenadamente
            images = sorted([f for f in image_dir.glob("*.png")])
            for idx, image_path in enumerate(images, start=1):
                # Ruta relativa sin extensión
                rel_path = image_path.relative_to(BASE_PATH).with_suffix("")
                lines.append(f"{rel_path} {idx} {CAMERA_LABEL}")

        # Guardar archivo .txt
        output_file = OUTPUT_DIR / f"{split}_files_hamlyn.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"[OK] Generado: {output_file} ({len(lines)} líneas)")

if __name__ == "__main__":
    generate_split_txts()
