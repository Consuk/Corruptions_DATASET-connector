
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import wandb


def resolve_seq_path(folder: str) -> str:
    """Given 'rectified05' or 'rectified05/rectified05', return the sequence root path."""
    norm = folder.strip("/")

    parts = norm.split("/")
    # If they already pass rectifiedXX/rectifiedXX, keep last 2
    if len(parts) >= 2 and parts[-1].startswith("rectified") and parts[-2].startswith("rectified"):
        return os.path.join(parts[-2], parts[-1])
    # If single token 'rectified05' -> rectified05/rectified05
    if len(parts) == 1:
        return os.path.join(parts[0], parts[0])
    # Otherwise, best effort: last 2 components
    return os.path.join(parts[-2], parts[-1])


def resolve_image_base(folder: str, side: str) -> str:
    """
    If folder already contains image01/image02 -> keep.
    Else map side: 'r' -> image02, else -> image01.
    """
    norm = folder.strip("/")
    parts = norm.split("/")

    if any(p == "image01" or p == "image02" for p in parts):
        return norm  # already points to the image folder

    seq_path = resolve_seq_path(norm)
    cam = "image02" if (side and side.lower().startswith("r")) else "image01"
    return os.path.join(seq_path, cam)


def find_frame_file(root: str, base: str, frame_id: int, exts) -> str:
    fname = f"{frame_id:010d}"
    for ext in exts:
        p = os.path.join(root, base, fname + ext)
        if os.path.isfile(p):
            return p
    return None


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Make a nice visualization for depth: clip to p1-p99 on nonzero values."""
    d = depth.astype(np.float32)
    v = d[d > 0]
    if v.size == 0:
        vis = np.zeros_like(d, dtype=np.float32)
        return vis
    p1, p99 = np.percentile(v, [1, 99]).astype(np.float32)
    if p99 <= p1:
        p1, p99 = float(v.min()), float(v.max())
        if p99 <= p1:
            return np.zeros_like(d, dtype=np.float32)
    vis = np.clip(d, p1, p99)
    # normalize to 0..1 for wandb
    vis = (vis - p1) / (p99 - p1 + 1e-6)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Hamlyn root, e.g. /workspace/datasets/hamlyn/Hamlyn")
    ap.add_argument("--split_file", required=True, help="Path to test_files2.txt")
    ap.add_argument("--project", default="hamlyn-test-images")
    ap.add_argument("--run_name", default="test_files2_rgb_log")
    ap.add_argument("--max_images", type=int, default=200, help="How many images to log (avoid huge runs). Use 0 for ALL.")
    ap.add_argument("--stride", type=int, default=1, help="Log every Nth sample")
    ap.add_argument("--start", type=int, default=0, help="Start index in split file")
    ap.add_argument("--gt_npz", default=None, help="Optional: path to test_files2_gt_depths.npz to log depth too")
    ap.add_argument("--log_depth", action="store_true", help="If set, logs gt depth (requires --gt_npz)")
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    print(f"Loaded split lines: {len(lines)} from {args.split_file}")

    gt = None
    if args.log_depth:
        if not args.gt_npz:
            raise ValueError("--log_depth requires --gt_npz")
        gt = np.load(args.gt_npz, allow_pickle=True)["data"]
        gt = list(gt) if isinstance(gt, np.ndarray) and gt.dtype == object else gt
        if len(gt) != len(lines):
            print(f"[WARN] gt maps ({len(gt)}) != split lines ({len(lines)}). Depth logging may misalign.")

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    table_cols = ["idx", "line", "image_path", "rgb"]
    if args.log_depth:
        table_cols += ["gt_depth_vis"]

    table = wandb.Table(columns=table_cols)

    exts = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]

    count = 0
    for idx in range(args.start, len(lines), max(1, args.stride)):
        if args.max_images and count >= args.max_images:
            break

        parts = lines[idx].split()
        if len(parts) < 2:
            print(f"[SKIP] bad line idx={idx}: {lines[idx]}")
            continue

        folder = parts[0]
        frame_id = int(parts[1])
        side = parts[2] if len(parts) > 2 else "l"

        img_base = resolve_image_base(folder, side)
        img_path = find_frame_file(args.data_path, img_base, frame_id, exts)

        if img_path is None:
            print(f"[MISS] idx={idx} | {lines[idx]} | tried base={os.path.join(args.data_path, img_base)}")
            continue

        try:
            rgb = load_rgb(img_path)
        except Exception as e:
            print(f"[ERR] idx={idx} reading {img_path}: {e}")
            continue

        caption = f"idx={idx} | {lines[idx]} | {img_base} | {os.path.basename(img_path)}"
        rgb_w = wandb.Image(rgb, caption=caption)

        row = [idx, lines[idx], img_path, rgb_w]

        if args.log_depth and gt is not None and idx < len(gt):
            d = gt[idx]
            dvis = depth_to_vis(d)
            # wandb.Image soporta floats 0..1
            row.append(wandb.Image(dvis, caption=f"GT depth (vis) | {caption}"))

        table.add_data(*row)
        count += 1

        if count % 25 == 0:
            wandb.log({"test_files2_samples": table})
            print(f"Logged {count} samples...")

    wandb.log({"test_files2_samples": table})
    wandb.finish()
    print(f"Done. Logged samples: {count}")


if __name__ == "__main__":
    main()

  