# generate_corruptions.py
import os
import argparse
from PIL import Image
import numpy as np
from endoscopycorruptions import corrupt, get_corruption_names
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate endoscopy corruptions for SCARED or Hamlyn datasets"
    )

    p.add_argument(
        "--dataset_type",
        type=str,
        choices=["scared", "hamlyn"],
        default="scared",
        help="Which dataset layout to use (default: scared)."
    )

    p.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root path of the dataset to corrupt."
    )

    p.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root path where corrupted dataset will be saved."
    )

    p.add_argument(
        "--corruptions",
        type=str,
        default="all",
        help="Comma-separated corruption names or 'all'."
    )

    p.add_argument(
        "--severities",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated severities, e.g. '1,2,3,4,5'."
    )

    p.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated extensions to include (overrides defaults). Example: '.jpg,.png'"
    )

    # --- Hamlyn-only options ---
    p.add_argument(
        "--hamlyn_only_rectified01_images",
        action="store_true",
        help="If set and dataset_type=hamlyn, only corrupt rectified01/rectified01/image01 and image02."
    )

    p.add_argument(
        "--hamlyn_image_dirs",
        type=str,
        default="image01,image02",
        help="Hamlyn image dirs to corrupt when hamlyn_only_rectified01_images is set. Default: image01,image02"
    )

    return p.parse_args()


def get_corruption_list(corruptions_arg):
    all_corrs = get_corruption_names()
    if corruptions_arg == "all":
        return all_corrs

    req = [c.strip() for c in corruptions_arg.split(",") if c.strip()]
    valid = [c for c in req if c in all_corrs]
    missing = set(req) - set(valid)

    if missing:
        print(f"[WARN] corrupciones no reconocidas e ignoradas: {sorted(missing)}")

    return valid


def get_extensions(args, default_exts):
    if args.extensions is None:
        return default_exts
    exts = tuple(
        e.strip().lower() for e in args.extensions.split(",") if e.strip()
    )
    return exts


# ======================
# MODO SCARED (igual que antes)
# Estructura esperada:
# input_root/
#   datasetX/
#     keyframeY/
#       data/
#         *.jpg
# ======================
def generate_scared(args, corruption_types, severities, extensions):
    input_root = args.input_root
    output_root = args.output_root

    for dataset in sorted(os.listdir(input_root)):
        dataset_path = os.path.join(input_root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for keyframe in sorted(os.listdir(dataset_path)):
            keyframe_path = os.path.join(dataset_path, keyframe)
            data_path = os.path.join(keyframe_path, "data")
            if not os.path.isdir(data_path):
                continue

            image_filenames = [
                f for f in os.listdir(data_path)
                if f.lower().endswith(extensions)
            ]

            for img_name in tqdm(image_filenames, desc=f"SCARED {dataset}/{keyframe}"):
                img_path = os.path.join(data_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_np = np.asarray(img)

                    for corr in corruption_types:
                        for sev in severities:
                            try:
                                img_corr = corrupt(
                                    img_np,
                                    corruption_name=corr,
                                    severity=sev
                                )
                            except Exception as e:
                                print(f"[WARN] {img_name} - {corr} sev{sev}: {e}")
                                continue

                            out_dir = os.path.join(
                                output_root,
                                corr,
                                f"severity_{sev}",
                                dataset,
                                keyframe,
                                "data"
                            )
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, img_name)

                            Image.fromarray(img_corr).save(out_path)

                except Exception as e:
                    print(f"[ERROR] cargando {img_path}: {e}")


# ======================
# MODO HAMLYN
# Estructura típica:
# input_root/Hamlyn/
#   rectifiedXX/rectifiedXX/
#     image01/*.png   (RGB)
#     image02/*.png   (RGB)
#     depth01/*.png   (NO corromper)
#     depth02/*.png   (NO corromper)
#
# Estrategia:
# - por default: recorre todo input_root y toma imágenes (png/jpg/jpeg),
#   saltando depth/disp/gt/mask.
# - si --hamlyn_only_rectified01_images:
#   solo toma rectified01/rectified01/image01,image02.
# - mantiene la ruta relativa en output_root.
# ======================
def generate_hamlyn(args, corruption_types, severities, extensions):
    input_root = args.input_root
    output_root = args.output_root

    # Siempre evitamos depth/disp/gt/mask
    skip_keywords = ("depth", "disp", "gt", "mask")

    def should_skip_dir(path):
        low = path.lower()
        return any(k in low for k in skip_keywords)

    image_paths = []

    if args.hamlyn_only_rectified01_images:
        # Solo rectified01/rectified01/image01,image02 (o lo que se ponga en hamlyn_image_dirs)
        image_dirs = [d.strip() for d in args.hamlyn_image_dirs.split(",") if d.strip()]
        allowed_roots = [
            os.path.join(input_root, "rectified01", "rectified01", d)
            for d in image_dirs
        ]

        for ar in allowed_roots:
            if not os.path.isdir(ar):
                print(f"[WARN] Hamlyn allowed root no existe: {ar}")
                continue

            for root, dirs, files in os.walk(ar):
                if should_skip_dir(root):
                    continue
                for f in files:
                    if f.lower().endswith(extensions):
                        image_paths.append(os.path.join(root, f))
    else:
        # Comportamiento general: todas las imágenes excepto depth
        for root, dirs, files in os.walk(input_root):
            if should_skip_dir(root):
                continue
            for f in files:
                if f.lower().endswith(extensions):
                    image_paths.append(os.path.join(root, f))

    print(f"[INFO] Hamlyn: encontré {len(image_paths)} imágenes para corromper.")

    for img_path in tqdm(image_paths, desc="HAMLYN corrompiendo"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.asarray(img)

            # Ruta relativa respecto al root de entrada
            rel_path = os.path.relpath(img_path, input_root)
            rel_dir = os.path.dirname(rel_path)
            img_name = os.path.basename(rel_path)

            for corr in corruption_types:
                for sev in severities:
                    try:
                        img_corr = corrupt(
                            img_np,
                            corruption_name=corr,
                            severity=sev
                        )
                    except Exception as e:
                        print(f"[WARN] {img_name} - {corr} sev{sev}: {e}")
                        continue

                    out_dir = os.path.join(
                        output_root,
                        corr,
                        f"severity_{sev}",
                        rel_dir
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, img_name)

                    Image.fromarray(img_corr).save(out_path)

        except Exception as e:
            print(f"[ERROR] cargando {img_path}: {e}")


def main():
    args = parse_args()

    corruption_types = get_corruption_list(args.corruptions)
    severities = [int(s) for s in args.severities.split(",") if s.strip()]

    if args.dataset_type == "scared":
        default_exts = (".jpg",)  # igual que antes
    else:
        default_exts = (".png", ".jpg", ".jpeg")  # hamlyn usa png

    extensions = get_extensions(args, default_exts)

    print("[INFO] dataset_type:", args.dataset_type)
    print("[INFO] input_root:", args.input_root)
    print("[INFO] output_root:", args.output_root)
    print("[INFO] corruptions:", corruption_types)
    print("[INFO] severities:", severities)
    print("[INFO] extensions:", extensions)

    os.makedirs(args.output_root, exist_ok=True)

    if args.dataset_type == "scared":
        generate_scared(args, corruption_types, severities, extensions)
    else:
        generate_hamlyn(args, corruption_types, severities, extensions)


if __name__ == "__main__":
    main()
