import os
from pathlib import Path

# Ruta base al dataset Hamlyn (carpeta que contiene rectified01, rectified04, etc.)
BASE_PATH = Path("/workspace/datasets/hamlyn/Hamlyn")
CAMERA_DIR = "image01"     # cámara izquierda
CAMERA_LABEL = "l"

# Proporciones del dataset SCARED (para replicar ratios)
TOTAL_SCARED = 15351 + 1705 + 551
SPLIT_RATIOS = {
    "train": 15351 / TOTAL_SCARED,
    "val":   1705  / TOTAL_SCARED,
    "test":  551   / TOTAL_SCARED,
}

# Carpeta donde se guardan los .txt de splits
OUTPUT_DIR = BASE_PATH.parent / "splits"
OUTPUT_DIR.mkdir(exist_ok=True)


def count_images_per_sequence():
    """
    Cuenta cuántas imágenes .jpg hay por secuencia rectifiedXX/rectifiedXX/image01.
    Devuelve un dict { 'rectified08': 2646, ... } ordenado de mayor a menor.
    """
    counts = {}
    for rectified in sorted(BASE_PATH.glob("rectified*")):
        seq_name = rectified.name  # p.ej. 'rectified08'
        img_dir = rectified / rectified.name / CAMERA_DIR  # rectified08/rectified08/image01
        if not img_dir.exists():
            continue
        num_imgs = len(list(img_dir.glob("*.jpg")))
        counts[seq_name] = num_imgs

    # ordenar de mayor a menor número de imágenes
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def assign_sequences_proportionally(image_counts):
    """
    Asigna secuencias completas a train/val/test intentando respetar
    los ratios de SCARED.
    """
    total_imgs = sum(image_counts.values())
    split_targets = {
        "train": round(SPLIT_RATIOS["train"] * total_imgs),
        "val":   round(SPLIT_RATIOS["val"]   * total_imgs),
        "test":  total_imgs,  # se ajusta luego
    }
    split_targets["test"] -= split_targets["train"] + split_targets["val"]

    splits = {"train": [], "val": [], "test": []}
    totals = {"train": 0, "val": 0, "test": 0}

    for seq, count in image_counts.items():
        remaining = {k: split_targets[k] - totals[k] for k in totals}
        split = max(remaining, key=remaining.get)
        splits[split].append(seq)
        totals[split] += count

    return splits


def generate_txts_from_splits(splits):
    """
    Genera train_files_hamlyn.txt, val_files_hamlyn.txt, test_files_hamlyn.txt
    con líneas del tipo:

        rectified11/rectified11/image01 812 l

    donde:
      - `folder` = rectifiedXX/rectifiedXX/image01
      - `frame_idx` = entero obtenido del nombre del archivo, e.g. 0000000812.jpg -> 812

    Se saltan el primer y último frame de cada secuencia para que siempre existan
    vecinos (frame_idx-1 y frame_idx+1) al hacer training monocular con [-1,0,1].
    """
    for split, seqs in splits.items():
        lines = []

        for seq in seqs:
            img_dir = BASE_PATH / seq / seq / CAMERA_DIR
            if not img_dir.exists():
                continue

            images = sorted(img_dir.glob("*.jpg"))
            if len(images) < 3:
                # muy pocas imágenes para usar vecinos
                continue

            # usamos sólo los frames "centrales": 1..len-2
            for center_pos in range(1, len(images) - 1):
                img = images[center_pos]
                # Carpeta base sin el nombre del archivo: rectifiedXX/rectifiedXX/image01
                folder = img.parent.relative_to(BASE_PATH)  # Path -> relativo a BASE_PATH

                # Nombre del archivo sin extensión: '0000000812'
                stem = img.stem
                try:
                    frame_idx = int(stem)
                except ValueError:
                    # Fallback súper raro, pero por seguridad
                    frame_idx = center_pos

                # Guardar usando POSIX paths (con '/')
                lines.append(f"{folder.as_posix()} {frame_idx} {CAMERA_LABEL}")

        output_file = OUTPUT_DIR / f"{split}_files_hamlyn.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        print(f"[✓] {output_file.name}: {len(lines)} líneas.")


def validate_split_file(split_file: Path):
    """
    Valida que cada línea del split corresponda a un archivo real,
    usando EXACTAMENTE la misma lógica que el dataloader de EndoDAC
    para Hamlyn:

        <BASE_PATH> / folder / f"{frame_idx:010d}.jpg"
    """
    print(f"\nValidating {split_file} ...")
    missing = 0
    total = 0

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print("  [WARN] línea rara:", line)
                continue

            folder_str, idx_str, _ = parts
            try:
                frame_idx = int(idx_str)
            except ValueError:
                print("  [WARN] frame_idx no entero en línea:", line)
                continue

            img_path = BASE_PATH / folder_str / f"{frame_idx:010d}.jpg"
            total += 1
            if not img_path.exists():
                missing += 1
                print("  MISSING:", img_path)

    print(f"Checked {total} lines. Missing files: {missing}")


if __name__ == "__main__":
    print("Contando imágenes por secuencia...")
    image_counts = count_images_per_sequence()

    print("Asignando secuencias proporcionalmente...")
    splits = assign_sequences_proportionally(image_counts)
    for s, seqs in splits.items():
        print(f"{s.upper()} ({len(seqs)} sec):", seqs)

    print("Generando archivos .txt...")
    generate_txts_from_splits(splits)

    # Validamos con la misma lógica que usa EndoDAC
    for name in ["train", "val", "test"]:
        validate_split_file(OUTPUT_DIR / f"{name}_files_hamlyn.txt")
