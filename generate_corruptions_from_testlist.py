import argparse
import os

import numpy as np
from PIL import Image
from endoscopycorruptions import corrupt, get_corruption_names
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate endoscopy corruptions from a Monodepth-style split file. "
            "Built for C3VD layout (e.g. test/<sequence>/<frame>_color.png)."
        )
    )
    parser.add_argument(
        "--test_list",
        type=str,
        required=True,
        help="Path to split file, e.g. Monodepth2/splits/c3vd/test_files.txt",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Dataset root, e.g. /workspace/datasets/c3vd_consuk_undist",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root, e.g. /workspace/datasets/c3vd_corrupted",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Split directory name under input_root when needed (default: test).",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        default="all",
        help="Comma-separated corruption names or 'all'.",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated severities.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of split lines to process (0 = all).",
    )
    return parser.parse_args()


def parse_corruptions(corruptions_arg: str):
    all_corrs = get_corruption_names()
    if corruptions_arg == "all":
        return all_corrs

    requested = [c.strip() for c in corruptions_arg.split(",") if c.strip()]
    selected = [c for c in requested if c in all_corrs]
    missing = sorted(set(requested) - set(selected))

    if missing:
        print(f"[WARN] Unknown corruptions ignored: {missing}")

    if not selected:
        raise ValueError("No valid corruption names were selected.")

    return selected


def normalize_frame_token(token: str):
    base = os.path.splitext(os.path.basename(token))[0]
    if base.isdigit():
        return str(int(base)), f"{int(base):04d}"
    return base, base


def candidate_dirs(input_root: str, split_name: str, rel_path: str):
    rel = rel_path.strip("/").replace("\\", "/")
    rel_no_split = rel[len(split_name) + 1 :] if rel.startswith(f"{split_name}/") else rel

    cands = [
        rel,
        rel_no_split,
        f"{split_name}/{rel_no_split}",
    ]

    unique = []
    for c in cands:
        c = c.strip("/")
        if c and c not in unique:
            unique.append(c)

    return [os.path.join(input_root, c) for c in unique], unique


def candidate_filenames(frame_raw: str, frame_4d: str):
    names = [
        f"{frame_4d}_color.png",
        f"{frame_raw}_color.png",
        f"{frame_4d}.png",
        f"{frame_raw}.png",
        f"{frame_raw}.jpg",
        f"{frame_raw}.jpeg",
    ]
    unique = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


def resolve_c3vd_path(input_root: str, split_name: str, rel_path: str, frame_token: str):
    frame_raw, frame_4d = normalize_frame_token(frame_token)
    dir_paths_abs, dir_paths_rel = candidate_dirs(input_root, split_name, rel_path)
    names = candidate_filenames(frame_raw, frame_4d)

    for d_abs, d_rel in zip(dir_paths_abs, dir_paths_rel):
        for name in names:
            p = os.path.join(d_abs, name)
            if os.path.isfile(p):
                return p, d_rel, os.path.splitext(name)[0]

    return None, None, None


def load_split(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            lines.append(parts)
    return lines


def main():
    args = parse_args()

    split_lines = load_split(args.test_list)
    if args.limit > 0:
        split_lines = split_lines[: args.limit]

    severities = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    corruption_types = parse_corruptions(args.corruptions)

    print(f"[INFO] split lines: {len(split_lines)}")
    print(f"[INFO] input_root: {args.input_root}")
    print(f"[INFO] output_root: {args.output_root}")
    print(f"[INFO] split_name: {args.split_name}")
    print(f"[INFO] corruptions: {corruption_types}")
    print(f"[INFO] severities: {severities}")

    os.makedirs(args.output_root, exist_ok=True)

    found = 0
    missing = 0
    load_errors = 0
    save_errors = 0

    for parts in tqdm(split_lines, desc="Generating corruptions"):
        rel_path, frame_token = parts[0], parts[1]
        img_path, out_rel, stem = resolve_c3vd_path(
            input_root=args.input_root,
            split_name=args.split_name,
            rel_path=rel_path,
            frame_token=frame_token,
        )

        if img_path is None:
            missing += 1
            print(f"[MISS] {rel_path} {frame_token}")
            continue

        found += 1

        try:
            img_np = np.asarray(Image.open(img_path).convert("RGB"))
        except Exception as exc:
            load_errors += 1
            print(f"[LOAD-ERROR] {img_path}: {exc}")
            continue

        for corr in corruption_types:
            for sev in severities:
                try:
                    img_corr = corrupt(img_np, corruption_name=corr, severity=sev)
                    out_dir = os.path.join(args.output_root, corr, f"severity_{sev}", out_rel)
                    os.makedirs(out_dir, exist_ok=True)

                    out_path = os.path.join(out_dir, f"{stem}.png")
                    Image.fromarray(img_corr).save(out_path)
                except Exception as exc:
                    save_errors += 1
                    print(f"[CORRUPT-ERROR] {corr} s{sev} | {img_path}: {exc}")

    print("\n[SUMMARY]")
    print(f"  total split lines : {len(split_lines)}")
    print(f"  resolved images   : {found}")
    print(f"  missing images    : {missing}")
    print(f"  load errors       : {load_errors}")
    print(f"  corruption errors : {save_errors}")


if __name__ == "__main__":
    main()

