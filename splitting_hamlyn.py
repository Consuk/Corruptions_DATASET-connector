import os
from pathlib import Path
from collections import defaultdict

# Ruta al dataset
BASE_PATH = Path("/workspace/datasets/hamlyn/Hamlyn")
CAMERA_DIR = "image01"
CAMERA_LABEL = "l"

# Proporciones del dataset SCARED
TOTAL_SCARED = 15351 + 1705 + 551
SPLIT_RATIOS = {
    "train": 15351 / TOTAL_SCARED,
    "val": 1705 / TOTAL_SCARED,
    "test": 551 / TOTAL_SCARED,
}

OUTPUT_DIR = BASE_PATH.parent / "splits"
OUTPUT_DIR.mkdir(exist_ok=True)

def count_images_per_sequence():
    counts = {}
    for rectified in sorted(BASE_PATH.glob("rectified*")):
        seq_name = rectified.name
        img_dir = rectified / rectified.name / CAMERA_DIR
        if not img_dir.exists():
            continue
        num_imgs = len(list(img_dir.glob("*.jpg")))
        counts[seq_name] = num_imgs
    return dict(sorted(counts.items(), key=lambda x: -x[1]))  # mayor a menor

def assign_sequences_proportionally(image_counts):
    total_imgs = sum(image_counts.values())
    split_targets = {
        "train": round(SPLIT_RATIOS["train"] * total_imgs),
        "val": round(SPLIT_RATIOS["val"] * total_imgs),
        "test": total_imgs,  # se ajusta luego
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
    for split, seqs in splits.items():
        lines = []
        for seq in seqs:
            img_dir = BASE_PATH / seq / seq / CAMERA_DIR
            if not img_dir.exists():
                continue
            images = sorted(img_dir.glob("*.jpg"))
            for idx, img in enumerate(images, start=1):
                rel_path = img.relative_to(BASE_PATH).with_suffix("")
                lines.append(f"{rel_path} {idx} {CAMERA_LABEL}")

        output_file = OUTPUT_DIR / f"{split}_files_hamlyn.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"[✓] {output_file.name}: {len(lines)} líneas.")

if __name__ == "__main__":
    print("Contando imágenes por secuencia...")
    image_counts = count_images_per_sequence()
    print("Asignando secuencias proporcionalmente...")
    splits = assign_sequences_proportionally(image_counts)
    for s, seqs in splits.items():
        print(f"{s.upper()} ({len(seqs)} sec):", seqs)
    print("Generando archivos .txt...")
    generate_txts_from_splits(splits)
