# log_to_wandb.py
import os
import random
import argparse
import wandb
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Log corrupted examples to Weights & Biases (SCARED or Hamlyn)."
    )

    p.add_argument(
        "--dataset_type",
        type=str,
        choices=["scared", "hamlyn"],
        default="scared",
        help="Dataset layout type (default: scared)."
    )

    p.add_argument(
        "--corrupted_root",
        type=str,
        required=True,
        help="Root folder with corruptions (e.g., /.../scared_corruptions or /.../hamlyn_corruptions)."
    )

    p.add_argument(
        "--examples_per_combo",
        type=int,
        default=3,
        help="How many images to log per corruption/severity combo."
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling."
    )

    p.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated extensions to include. Example: '.jpg,.png'. If None, uses defaults per dataset_type."
    )

    p.add_argument(
        "--project",
        type=str,
        default="endoscopy-corruption-viewer",
        help="W&B project name."
    )

    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional W&B run name."
    )

    return p.parse_args()


def get_extensions(args, default_exts):
    if args.extensions is None:
        return default_exts
    return tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())


def main():
    args = parse_args()

    if args.dataset_type == "scared":
        default_exts = (".jpg",)
        skip_keywords = ()  # no hace falta filtrar más
    else:
        default_exts = (".png", ".jpg", ".jpeg")
        # por seguridad, si hubiera algo raro dentro del output
        skip_keywords = ("depth", "disp", "gt", "mask")

    extensions = get_extensions(args, default_exts)
    random.seed(args.seed)

    run_name = args.run_name or f"{args.dataset_type}-visual-inspection"

    wandb.init(
        project=args.project,
        name=run_name,
        config={
            "dataset_type": args.dataset_type,
            "corrupted_root": args.corrupted_root,
            "examples_per_combo": args.examples_per_combo,
            "seed": args.seed,
            "extensions": extensions,
        },
    )

    CORRUPTED_ROOT = args.corrupted_root

    def should_skip_path(path):
        low = path.lower()
        return any(k in low for k in skip_keywords)

    # Recorremos corrupciones
    for corruption in sorted(os.listdir(CORRUPTED_ROOT)):
        corr_path = os.path.join(CORRUPTED_ROOT, corruption)
        if not os.path.isdir(corr_path):
            continue

        # severity folders (e.g., severity_1 ... severity_5)
        for severity in sorted(os.listdir(corr_path)):
            sev_path = os.path.join(corr_path, severity)
            if not os.path.isdir(sev_path):
                continue

            collected = []

            # caminar subdirectorios y tomar muestras
            for root, _, files in os.walk(sev_path):
                if should_skip_path(root):
                    continue

                image_files = [
                    f for f in files if f.lower().endswith(extensions)
                ]
                random.shuffle(image_files)

                for img_file in image_files:
                    if len(collected) >= args.examples_per_combo:
                        break
                    img_path = os.path.join(root, img_file)
                    if should_skip_path(img_path):
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        caption = f"{corruption}/{severity} - {os.path.relpath(img_path, CORRUPTED_ROOT)}"
                        collected.append(wandb.Image(img, caption=caption))
                    except Exception as e:
                        print(f"[WARN] Error loading {img_path}: {e}")

                if len(collected) >= args.examples_per_combo:
                    break

            if collected:
                wandb.log({f"{corruption}/{severity}": collected})

    wandb.finish()


if __name__ == "__main__":
    main()
