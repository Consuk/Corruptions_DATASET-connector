from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one GT depth map from a npz/npy file for visual inspection."
    )
    parser.add_argument("--gt_depths_file", required=True)
    parser.add_argument("--gt_depths_key", default="data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="./new_qualitative_images/gt_depth_preview")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--normalize_low", type=float, default=2.0)
    parser.add_argument("--normalize_high", type=float, default=98.0)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument(
        "--cell_width",
        type=int,
        default=0,
        help="Optional paper-grid cell width for an extra cover-cropped preview.",
    )
    parser.add_argument(
        "--cell_height",
        type=int,
        default=0,
        help="Optional paper-grid cell height for an extra cover-cropped preview.",
    )
    return parser.parse_args()


def load_depths(path: Path, key: str) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        selected_key = key if key in data.files else data.files[0]
        return data[selected_key]
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    raise ValueError(f"Expected .npz or .npy GT file, got: {path}")


def depth_at(depths: Any, index: int) -> np.ndarray:
    if isinstance(depths, np.ndarray) and depths.ndim <= 2:
        if index not in (0, -1):
            raise IndexError(f"Single-map GT file cannot use index {index}")
        return np.squeeze(depths).astype(np.float32)
    return np.squeeze(np.asarray(depths[index])).astype(np.float32)


def valid_mask(depth: np.ndarray, min_depth: Optional[float], max_depth: Optional[float]) -> np.ndarray:
    mask = np.isfinite(depth)
    if min_depth is not None:
        mask &= depth > min_depth
    else:
        mask &= depth > 0
    if max_depth is not None:
        mask &= depth < max_depth
    return mask


def safe_percentiles(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize(values: np.ndarray, mask: np.ndarray, cmap_name: str, low: float, high: float) -> Image.Image:
    visual = np.full_like(values, np.nan, dtype=np.float32)
    visual[mask] = values[mask]
    lo, hi = safe_percentiles(visual, low, high)
    norm = (np.clip(visual, lo, hi) - lo) / (hi - lo + 1e-8)
    norm[~np.isfinite(norm)] = 0.0

    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgb = (cmap(np.clip(norm, 0.0, 1.0))[..., :3] * 255.0).astype(np.uint8)
    except Exception:
        x = np.clip(norm[..., None], 0.0, 1.0)
        stops = np.array(
            [
                [0, 0, 4],
                [80, 18, 123],
                [182, 54, 121],
                [251, 136, 97],
                [252, 253, 191],
            ],
            dtype=np.float32,
        )
        pos = np.clip(x * (len(stops) - 1), 0, len(stops) - 1 - 1e-6)
        idx = np.floor(pos).astype(np.int32)
        frac = pos - idx
        rgb = (stops[idx[..., 0]] * (1.0 - frac) + stops[idx[..., 0] + 1] * frac).astype(
            np.uint8
        )

    rgb[~mask] = 0
    return Image.fromarray(rgb, mode="RGB")


def resize_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized = image.resize(
        (max(1, round(src_w * scale)), max(1, round(src_h * scale))),
        Image.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def stats(depth: np.ndarray, mask: np.ndarray) -> dict:
    valid = depth[mask]
    out = {
        "shape": list(depth.shape),
        "valid_fraction": float(mask.mean()),
        "valid_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
    }
    if valid.size:
        out.update(
            {
                "min": float(np.min(valid)),
                "p1": float(np.percentile(valid, 1)),
                "p50": float(np.percentile(valid, 50)),
                "p99": float(np.percentile(valid, 99)),
                "max": float(np.max(valid)),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_depths_file).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{gt_path.stem}_idx{args.index}"

    depth = depth_at(load_depths(gt_path, args.gt_depths_key), args.index)
    mask = valid_mask(depth, args.min_depth, args.max_depth)

    depth_img = colorize(depth, mask, args.cmap, args.normalize_low, args.normalize_high)
    inv = np.full_like(depth, np.nan, dtype=np.float32)
    inv[mask] = 1.0 / np.maximum(depth[mask], 1e-8)
    inv_img = colorize(inv, mask, args.cmap, args.normalize_low, args.normalize_high)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")

    outputs = {
        "depth": output_dir / f"{prefix}_depth.png",
        "inverse_depth": output_dir / f"{prefix}_inverse_depth.png",
        "valid_mask": output_dir / f"{prefix}_valid_mask.png",
        "stats": output_dir / f"{prefix}_stats.json",
    }
    depth_img.save(outputs["depth"])
    inv_img.save(outputs["inverse_depth"])
    mask_img.save(outputs["valid_mask"])

    if args.cell_width > 0 and args.cell_height > 0:
        outputs["depth_cell"] = output_dir / f"{prefix}_depth_cell.png"
        outputs["inverse_depth_cell"] = output_dir / f"{prefix}_inverse_depth_cell.png"
        resize_cover(depth_img, args.cell_width, args.cell_height).save(outputs["depth_cell"])
        resize_cover(inv_img, args.cell_width, args.cell_height).save(outputs["inverse_depth_cell"])

    with open(outputs["stats"], "w", encoding="utf-8") as f:
        json.dump(stats(depth, mask), f, indent=2)

    print("Saved GT previews:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
