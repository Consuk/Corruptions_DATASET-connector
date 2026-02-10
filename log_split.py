import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb
import matplotlib.cm as cm

from utils import readlines
import datasets
import networks
from layers import disp_to_depth


def tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """
    t: (1,3,H,W) or (3,H,W), assumed in [0,1] after dataset preprocessing.
    Returns uint8 RGB HxWx3
    """
    if t.ndim == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0, 1)
    img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def sanitize_gt_depth(d: np.ndarray) -> np.ndarray:
    """Remove typical invalid sentinels."""
    d = d.astype(np.float32)
    d[(d == 65535.0) | (d < 0)] = 0.0
    return d


def percentile_normalize(x: np.ndarray, valid_mask: np.ndarray, p_lo=1, p_hi=99) -> np.ndarray:
    """
    Normalize x to [0,1] using percentiles computed on valid_mask pixels.
    Returns float32 HxW in [0,1]
    """
    if valid_mask is None:
        valid_mask = np.isfinite(x)

    v = x[valid_mask]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(v, [p_lo, p_hi]).astype(np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)

    y = np.clip(x, lo, hi)
    y = (y - lo) / (hi - lo + 1e-6)
    return y.astype(np.float32)


def colorize01(x01: np.ndarray, cmap_name="magma") -> np.ndarray:
    """
    x01: float32 HxW in [0,1]
    Returns uint8 RGB HxWx3
    """
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(x01, 0, 1))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def make_valid_mask(gt: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return (gt > min_depth) & (gt < max_depth)


def load_model(weights_folder: str, device: torch.device):
    # ResNet18 encoder + standard DepthDecoder (como evaluate_depth.py)
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    encoder_path = os.path.join(weights_folder, "encoder.pth")
    depth_path = os.path.join(weights_folder, "depth.pth")

    if not os.path.isfile(encoder_path) or not os.path.isfile(depth_path):
        raise FileNotFoundError(
            f"Expected encoder.pth and depth.pth in {weights_folder}. "
            f"Found encoder={os.path.isfile(encoder_path)} depth={os.path.isfile(depth_path)}"
        )

    loaded = torch.load(encoder_path, map_location=device)
    filtered = {k: v for k, v in loaded.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered)

    depth_decoder.load_state_dict(torch.load(depth_path, map_location=device))

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    return encoder, depth_decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split_file", required=True, help="test_files2.txt")
    ap.add_argument("--gt_npz", required=True, help="test_files2_gt_depths.npz (data=object array)")
    ap.add_argument("--weights_folder", required=True, help="folder with encoder.pth & depth.pth")
    ap.add_argument("--dataset", default="hamlyn")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--min_depth", type=float, default=1.0)
    ap.add_argument("--max_depth", type=float, default=300.0)

    ap.add_argument("--max_images", type=int, default=200, help="0 = all")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)

    ap.add_argument("--project", default="hamlyn-eval-visuals")
    ap.add_argument("--run_name", default="test_files2_rgb_gt_pred_error")
    ap.add_argument("--cmap", default="magma")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filenames = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    gt = np.load(args.gt_npz, allow_pickle=True)["data"]
    gt = list(gt) if isinstance(gt, np.ndarray) and gt.dtype == object else gt

    if len(gt) != len(filenames):
        print(f"[WARN] gt maps ({len(gt)}) != split lines ({len(filenames)}). (Ideal: same)")

    # dataset loader (Monodepth2-style)
    if args.dataset.lower() == "hamlyn":
        DatasetClass = datasets.HamlynDataset
    else:
        raise ValueError(f"Only hamlyn implemented in this logger. Got: {args.dataset}")

    dataset = DatasetClass(
        args.data_path,
        filenames,
        args.height,
        args.width,
        frame_idxs=[0],
        num_scales=4,
        is_train=False
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    encoder, depth_decoder = load_model(args.weights_folder, device)

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    cols = [
        "idx",
        "line",
        "rgb",
        "gt_depth",
        "pred_depth",
        "absrel_error",
        "valid_frac",
        "gt_med",
        "pred_med",
    ]
    table = wandb.Table(columns=cols)

    logged = 0
    with torch.no_grad():
        for i, inputs in enumerate(loader):
            if i < args.start:
                continue
            if (i - args.start) % max(1, args.stride) != 0:
                continue
            if args.max_images and logged >= args.max_images:
                break

            # RGB input tensor key (Monodepth2)
            color = inputs[("color", 0, 0)].to(device)  # (B,3,H,W)

            # Forward
            feats = encoder(color)
            outputs = depth_decoder(feats)
            disp = outputs[("disp", 0)]
            _, pred_depth_t = disp_to_depth(disp, args.min_depth, args.max_depth)  # (B,1,H,W)

            # Convert RGB
            rgb_u8 = tensor_to_uint8_rgb(color)

            # GT depth for this index
            if i >= len(gt):
                print(f"[WARN] No GT for idx={i}; skipping")
                continue
            gt_d = sanitize_gt_depth(gt[i])

            # Resize pred depth to GT shape (como evaluate_depth.py hace internamente)
            pred_depth = pred_depth_t[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_depth = cv2.resize(pred_depth, (gt_d.shape[1], gt_d.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Build mask
            m = make_valid_mask(gt_d, args.min_depth, args.max_depth)
            valid_frac = float(m.mean())

            # Stats
            gt_med = float(np.median(gt_d[m])) if m.any() else float("nan")
            pred_med = float(np.median(pred_depth[m])) if m.any() else float("nan")

            # Visualizations (clear)
            gt01 = percentile_normalize(gt_d, m, 1, 99)
            pred01 = percentile_normalize(pred_depth, m, 1, 99)

            gt_rgb = colorize01(gt01, args.cmap)
            pred_rgb = colorize01(pred01, args.cmap)

            # AbsRel error map: |pred-gt|/gt on valid mask
            err = np.zeros_like(gt_d, dtype=np.float32)
            if m.any():
                err[m] = np.abs(pred_depth[m] - gt_d[m]) / (gt_d[m] + 1e-6)
            # normalize error for display (clip to p99)
            err01 = percentile_normalize(err, m, 1, 99)
            err_rgb = colorize01(err01, "inferno")  # error se ve mejor con inferno

            caption = f"idx={i} | valid={valid_frac:.3f} | gt_med={gt_med:.2f} pred_med={pred_med:.2f} | {filenames[i]}"

            table.add_data(
                i,
                filenames[i],
                wandb.Image(rgb_u8, caption="RGB | " + caption),
                wandb.Image(gt_rgb, caption="GT depth (p1–p99) | " + caption),
                wandb.Image(pred_rgb, caption="Pred depth (p1–p99) | " + caption),
                wandb.Image(err_rgb, caption="AbsRel error (p1–p99) | " + caption),
                valid_frac,
                gt_med,
                pred_med
            )

            logged += 1
            if logged % 25 == 0:
                wandb.log({"test_files2_visuals": table})
                print(f"Logged {logged} samples...")

    wandb.log({"test_files2_visuals": table})
    wandb.finish()
    print("Done. Logged:", logged)


if __name__ == "__main__":
    main()
