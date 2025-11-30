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
            img_dir = BASE_PATH / seq / seq / CAMERA_DIR  # rectifiedXX/rectifiedXX/image01
            if not img_dir.exists():
                continue
            images = sorted(img_dir.glob("*.jpg"))  # 0000000000.jpg, ...

            for img in images:
                # Name without extension, e.g. "0000001805"
                stem = img.stem
                try:
                    frame_idx = int(stem)  # 1805
                except ValueError:
                    # If some weird file appears, skip it
                    print(f"[WARN] Skipping non-numeric file: {img}")
                    continue

                rel_folder = img.parent.relative_to(BASE_PATH)  # rectified08/rectified08/image01
                lines.append(f"{rel_folder} {frame_idx} {CAMERA_LABEL}")

        output_file = OUTPUT_DIR / f"{split}_files_hamlyn.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"[✓] {output_file.name}: {len(lines)} líneas.")


def validate_split_file(split_file, img_ext=".jpg"):
    print(f"\nValidating {split_file} ...")
    missing = 0
    total = 0

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split()
            if len(parts) != 3:
                print("[WARN] Malformed line:", line)
                continue

            folder, idx_str, side = parts
            try:
                idx = int(idx_str)
            except ValueError:
                print("[WARN] Non-integer index:", line)
                continue

            # Our new format: folder = rectifiedXX/rectifiedXX/image01
            # File = <BASE_PATH>/<folder>/<idx formatted as 10 digits>.jpg
            img_name = f"{idx:010d}{img_ext}"
            img_path = BASE_PATH / folder / img_name

            if not img_path.exists():
                missing += 1
                if missing <= 20:  # don't spam too much
                    print("  [MISSING]", img_path)

    print(f"Checked {total} lines. Missing files: {missing}")
    return missing

if __name__ == "__main__":
    print("Contando imágenes por secuencia...")
    image_counts = count_images_per_sequence()

    print("Asignando secuencias proporcionalmente...")
    splits = assign_sequences_proportionally(image_counts)
    for s, seqs in splits.items():
        print(f"{s.upper()} ({len(seqs)} sec):", seqs)

    print("Generando archivos .txt...")
    generate_txts_from_splits(splits)

    # ---- Validate the generated splits ----
    for split in ["train", "val", "test"]:
        split_file = OUTPUT_DIR / f"{split}_files_hamlyn.txt"
        validate_split_file(split_file, img_ext=".jpg")

