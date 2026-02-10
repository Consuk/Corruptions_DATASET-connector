import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import wandb
import matplotlib.cm as cm


def resolve_seq_path(seq_or_path: str) -> str:
    norm = seq_or_path.strip("/")
    parts = norm.split("/")

    # rectifiedXX/rectifiedXX
    if len(parts) >= 2 and parts[-2].startswith("rectified") and parts[-1].startswith("rectified"):
        return os.path.join(parts[-2], parts[-1])

    # rectifiedXX/rectifiedXX/image01
    if len(parts) >= 3 and parts[-3].startswith("rectified") and parts[-2].startswith("rectified"):
        return os.path.join(parts[-3], parts[-2])

    # rectifiedXX
    if len(parts) == 1:
        return os.path.join(parts[0], parts[0])

    return os.path.join(parts[-2], parts[-1])


def resolve_image_base(folder: str, side: str) -> str:
    norm = folder.strip("/")
    parts = norm.split("/")
    if "image01" in parts or "image02" in parts:
        return norm

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
    d[(d == 65535.0) | (d < 0)] = 0.0
    return d


def stretch_to_01(d: np.ndarray, mask: np.ndarray, p_lo=1, p_hi=99):
    """
    Normalize d to [0,1] using percentiles computed on mask.
    Returns (d01, p1, p99)
    """
    d = d.astype(np.float32)
    vals = d[mask]
    if vals.size == 0:
        return np.zeros_like(d, dtype=np.float32), None, None

    p1, p99 = np.percentile(vals, [p_lo, p_hi]).astype(np.float32)
    if p99 <= p1:
        p1, p99 = float(vals.min()), float(vals.max())
        if p99 <= p1:
            return np.zeros_like(d, dtype=np.float32), p1, p99

    x = np.clip(d, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-6)
    return x.astype(np.float32), float(p1), float(p99)


def colorize01(x01: np.ndarray, cmap_name="magma") -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(x01, 0, 1))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--project", default="hamlyn-split-logs")
    ap.add_argument("--run_name", default="test_files2_rgb_gt_overlay")

    ap.add_argument("--max_images", type=int, default=200, help="0 = todas")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)

    ap.add_argument("--log_gt", action="store_true")
    ap.add_argument("--gt_npz", default=None)

    ap.add_argument("--min_depth", type=float, default=1.0)
    ap.add_argument("--max_depth", type=float, default=300.0)

    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--overlay_alpha", type=float, default=0.55)

    ap.add_argument("--thumb_max_side", type=int, default=320,
                    help="Reduce tamaño para que NO haya scrollbars en la tabla. 0 = sin resize")
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

    cols = ["idx", "line", "rgb", "rgb_path"]
    if args.log_gt:
        cols += ["gt_depth_clear", "mask", "overlay", "valid_frac"]
    table = wandb.Table(columns=cols)

    logged = 0
    for idx in range(args.start, len(lines), max(1, args.stride)):
        if args.max_images and logged >= args.max_images:
            break

        parts = lines[idx].split()
        if len(parts) < 2:
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
        rgb_thumb = resize_max(rgb, args.thumb_max_side)

        caption = f"idx={idx} | {lines[idx]} | {img_base} | {os.path.basename(img_path)}"
        row = [idx, lines[idx], wandb.Image(rgb_thumb, caption=caption), img_path]

        if args.log_gt and gt is not None and idx < len(gt):
            d = sanitize_depth(gt[idx])

            # máscara "válida" como en eval: rango min..max
            m = (d > args.min_depth) & (d < args.max_depth)
            valid_frac = float(m.mean())

            d01, p1, p99 = stretch_to_01(d, m, 1, 99)
            dclr = colorize01(d01, args.cmap)

            # mask para visualizar
            mask_u8 = (m.astype(np.uint8) * 255)
            mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=2)

            # overlay depth sobre rgb (mismo tamaño que rgb)
            dclr_rs = cv2.resize(dclr, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            m_rs = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

            overlay = rgb.copy()
            a = float(np.clip(args.overlay_alpha, 0.0, 1.0))
            overlay[m_rs] = (overlay[m_rs] * (1.0 - a) + dclr_rs[m_rs] * a).astype(np.uint8)

            # thumbs
            dclr_thumb = resize_max(dclr, args.thumb_max_side)
            mask_thumb = resize_max(mask_rgb, args.thumb_max_side)
            overlay_thumb = resize_max(overlay, args.thumb_max_side)

            extra = f"valid={valid_frac:.3f}"
            if p1 is not None and p99 is not None:
                extra += f" | p1={p1:.1f} p99={p99:.1f}"
            gt_caption = f"GT clear | {extra} | {caption}"

            row += [
                wandb.Image(dclr_thumb, caption=gt_caption),
                wandb.Image(mask_thumb, caption=f"Mask (valid) | {extra} | {caption}"),
                wandb.Image(overlay_thumb, caption=f"Overlay (GT on RGB) | {extra} | {caption}"),
                valid_frac
            ]

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
