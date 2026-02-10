import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import wandb
import matplotlib.cm as cm


def resolve_seq_path(seq_or_path: str) -> str:
    """
    Input examples:
      - rectified05
      - rectified05/rectified05
      - rectified05/rectified05/image01  (lo normalizamos al root de secuencia)
    Output:
      - rectified05/rectified05
    """
    norm = seq_or_path.strip("/")

    parts = norm.split("/")
    # si ya viene rectifiedXX/rectifiedXX/...
    if len(parts) >= 2 and parts[-2].startswith("rectified") and parts[-1].startswith("rectified"):
        return os.path.join(parts[-2], parts[-1])

    # si viene rectifiedXX/rectifiedXX/image01
    if len(parts) >= 3 and parts[-3].startswith("rectified") and parts[-2].startswith("rectified"):
        return os.path.join(parts[-3], parts[-2])

    # si viene solo rectifiedXX
    if len(parts) == 1:
        return os.path.join(parts[0], parts[0])

    # fallback: últimos 2
    return os.path.join(parts[-2], parts[-1])


def resolve_image_base(folder: str, side: str) -> str:
    """
    folder puede venir como:
      - rectified05
      - rectified05/rectified05
      - rectified05/rectified05/image01
      - rectified05/rectified05/image02
    side:
      - 'l' -> image01
      - 'r' -> image02
    """
    norm = folder.strip("/")
    parts = norm.split("/")
    if "image01" in parts or "image02" in parts:
        return norm  # ya apunta a image folder

    seq_root = resolve_seq_path(norm)
    cam = "image02" if (side and side.lower().startswith("r")) else "image01"
    return os.path.join(seq_root, cam)


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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sanitize_depth(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    # valores inválidos típicos
    d[(d == 65535.0) | (d < 0)] = 0.0
    return d


def stretch_to_01(d: np.ndarray, mask: np.ndarray = None, p_lo=1, p_hi=99) -> np.ndarray:
    """
    Normaliza a [0,1] usando percentiles sobre pixeles válidos.
    """
    d = d.astype(np.float32)
    if mask is None:
        mask = d > 0
    vals = d[mask]
    if vals.size == 0:
        return np.zeros_like(d, dtype=np.float32)

    lo, hi = np.percentile(vals, [p_lo, p_hi]).astype(np.float32)
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros_like(d, dtype=np.float32)

    x = np.clip(d, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return x.astype(np.float32)


def colorize01(x01: np.ndarray, cmap_name="magma") -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(x01, 0, 1))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Ej: /workspace/datasets/hamlyn/Hamlyn")
    ap.add_argument("--split_file", required=True, help="Ej: /workspace/datasets/hamlyn/splits/test_files2.txt")
    ap.add_argument("--project", default="hamlyn-split-logs")
    ap.add_argument("--run_name", default="test_files2_rgb_gt")

    ap.add_argument("--max_images", type=int, default=200, help="0 = todas")
    ap.add_argument("--stride", type=int, default=5, help="loggea cada N samples")
    ap.add_argument("--start", type=int, default=0)

    ap.add_argument("--log_gt", action="store_true", help="Loggea GT depth si pasas --gt_npz")
    ap.add_argument("--gt_npz", default=None, help="Ej: /workspace/datasets/hamlyn/splits/test_files2_gt_depths.npz")

    ap.add_argument("--min_depth", type=float, default=1.0)
    ap.add_argument("--max_depth", type=float, default=300.0)

    ap.add_argument("--cmap", default="magma")
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(lines)} lines from {args.split_file}")

    gt = None
    if args.log_gt:
        if not args.gt_npz:
            raise ValueError("--log_gt requiere --gt_npz")
        gt = np.load(args.gt_npz, allow_pickle=True)["data"]
        gt = list(gt) if isinstance(gt, np.ndarray) and gt.dtype == object else gt
        print(f"Loaded GT maps: {len(gt)} from {args.gt_npz}")
        if len(gt) != len(lines):
            print(f"[WARN] GT count ({len(gt)}) != lines ({len(lines)}). Se loggea por índice.")

    exts = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    cols = ["idx", "line", "image_path", "rgb_clear"]
    if args.log_gt:
        cols += ["gt_depth_clear", "valid_frac"]
    table = wandb.Table(columns=cols)

    logged = 0
    for idx in range(args.start, len(lines), max(1, args.stride)):
        if args.max_images and logged >= args.max_images:
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
            print(f"[MISS] idx={idx} | {lines[idx]} | base={os.path.join(args.data_path, img_base)}")
            continue

        rgb = load_rgb(img_path)
        caption = f"idx={idx} | {lines[idx]} | {img_base} | {os.path.basename(img_path)}"
        rgb_w = wandb.Image(rgb, caption=caption)

        row = [idx, lines[idx], img_path, rgb_w]

        if args.log_gt and gt is not None and idx < len(gt):
            d = sanitize_depth(gt[idx])
            # máscara consistente con evaluación
            m = (d > args.min_depth) & (d < args.max_depth)
            valid_frac = float(m.mean())

            d01 = stretch_to_01(d, mask=m, p_lo=1, p_hi=99)  # mejora claridad
            dclr = colorize01(d01, args.cmap)
            row += [wandb.Image(dclr, caption=f"GT depth clear | valid={valid_frac:.3f} | {caption}"), valid_frac]

        table.add_data(*row)
        logged += 1

        if logged % 25 == 0:
            wandb.log({"test_files2_table": table})
            print(f"Logged {logged} rows...")

    wandb.log({"test_files2_table": table})
    wandb.finish()
    print("Done. Logged:", logged)


if __name__ == "__main__":
    main()
