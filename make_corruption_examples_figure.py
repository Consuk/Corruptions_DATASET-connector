from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_qualitative_corruption_grid import (
    data_root_for_severity,
    extension_list,
    pick_reference_image,
    resolve_row_image,
    severity_root,
)


DEFAULT_CORRUPTIONS = (
    "brightness",
    "darkness",
    "contrast",
    "fog",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_noise",
    "impulse_noise",
    "shot_noise",
    "iso_noise",
    "lens_distortion",
    "resolution_change",
    "specular_reflection",
    "color_changes",
)

DISPLAY_LABELS = {
    "brightness": "Brightness",
    "darkness": "Darkness",
    "contrast": "Contrast",
    "fog": "Fog",
    "defocus_blur": "Defocus Blur",
    "glass_blur": "Glass Blur",
    "motion_blur": "Motion Blur",
    "zoom_blur": "Zoom Blur",
    "gaussian_noise": "Gaussian\nNoise",
    "impulse_noise": "Impulse\nNoise",
    "shot_noise": "Shot Noise",
    "iso_noise": "Iso Noise",
    "lens_distortion": "Lens\nDistortion",
    "resolution_change": "Resolution\nChange",
    "specular_reflection": "Specular\nReflection",
    "color_changes": "Color\nChanges",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def display_label(corruption: str) -> str:
    return DISPLAY_LABELS.get(corruption.lower(), corruption.replace("_", " ").title())


def resize_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    resized = img.resize(
        (max(1, round(src_w * scale)), max(1, round(src_h * scale))),
        Image.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def draw_center_label(tile: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    tile = tile.convert("RGB")
    draw = ImageDraw.Draw(tile)
    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in bboxes]
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = (tile.height - total_h) // 2
    for line, width, line_h in zip(lines, widths, line_heights):
        x = (tile.width - width) // 2
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2)]:
            draw.text((x + dx, y + dy), line, fill=(55, 55, 55), font=font)
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += line_h + 4
    return tile


def wrapped_caption(caption: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    if not caption:
        return []
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    words = caption.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-style grid showing examples of image corruptions."
    )
    parser.add_argument("--corruptions_root", required=True)
    parser.add_argument("--output", default="corruption_examples_figure.png")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument(
        "--corruptions",
        default=",".join(DEFAULT_CORRUPTIONS),
        help="Comma-separated corruption folder names.",
    )
    parser.add_argument("--rel_image", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png")
    parser.add_argument("--nested_data_dir", default="endovis_data")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--tile_width", type=int, default=165)
    parser.add_argument("--tile_height", type=int, default=120)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--figure_margin", type=int, default=44)
    parser.add_argument("--label_font_size", type=int, default=18)
    parser.add_argument("--caption_font_size", type=int, default=16)
    parser.add_argument("--caption", default="")
    parser.add_argument("--save_metadata", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_key", default="corruption_examples")
    return parser.parse_args()


def log_to_wandb(args: argparse.Namespace, output_path: Path, metadata: dict) -> None:
    if not args.wandb_project:
        return
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is not installed; omit --wandb_project or install wandb.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_path.stem,
        config=metadata,
    )
    wandb.log({args.wandb_key: wandb.Image(str(output_path), caption=args.caption)})
    wandb.save(str(output_path))
    run.finish()


def main() -> None:
    args = parse_args()
    corruptions_root = Path(args.corruptions_root).expanduser()
    output_path = Path(args.output).expanduser()
    exts = extension_list(args.extensions)
    corruptions = [item.strip() for item in args.corruptions.split(",") if item.strip()]
    if not corruptions:
        raise ValueError("No corruptions were selected")

    first_data_root = data_root_for_severity(
        severity_root(corruptions_root, corruptions[0], args.severity),
        args.nested_data_dir,
    )
    _, reference_rel = pick_reference_image(
        root=first_data_root,
        rel_image=args.rel_image,
        split_file=args.split_file,
        split_index=args.split_index,
        exts=exts,
    )

    cols = max(1, args.cols)
    rows = (len(corruptions) + cols - 1) // cols
    grid_w = cols * args.tile_width + (cols - 1) * args.gap
    grid_h = rows * args.tile_height + (rows - 1) * args.gap
    caption_font = load_font(args.caption_font_size)
    caption_lines = wrapped_caption(args.caption, caption_font, grid_w)
    caption_h = 0
    if caption_lines:
        probe = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(probe)
        line_h = max(
            draw.textbbox((0, 0), line, font=caption_font)[3]
            - draw.textbbox((0, 0), line, font=caption_font)[1]
            for line in caption_lines
        )
        caption_h = 24 + len(caption_lines) * (line_h + 4)

    canvas_w = grid_w + 2 * args.figure_margin
    canvas_h = grid_h + 2 * args.figure_margin + caption_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    label_font = load_font(args.label_font_size, bold=True)
    grid_x = args.figure_margin
    grid_y = args.figure_margin

    metadata = {
        "corruptions_root": str(corruptions_root),
        "severity": args.severity,
        "reference_rel": reference_rel,
        "corruptions": corruptions,
        "rows": [],
    }

    for index, corruption in enumerate(corruptions):
        sev_root = severity_root(corruptions_root, corruption, args.severity)
        data_root = data_root_for_severity(sev_root, args.nested_data_dir)
        image_path, row_rel = resolve_row_image(
            data_root=data_root,
            rel_path=reference_rel,
            rel_image=args.rel_image,
            split_file=args.split_file,
            split_index=args.split_index,
            exts=exts,
        )
        tile = resize_cover(Image.open(image_path), (args.tile_width, args.tile_height))
        tile = draw_center_label(tile, display_label(corruption), label_font)
        row = index // cols
        col = index % cols
        x = grid_x + col * (args.tile_width + args.gap)
        y = grid_y + row * (args.tile_height + args.gap)
        canvas.paste(tile, (x, y))
        metadata["rows"].append(
            {
                "corruption": corruption,
                "label": display_label(corruption),
                "image_path": str(image_path),
                "relative_path": row_rel,
            }
        )

    if caption_lines:
        draw = ImageDraw.Draw(canvas)
        y = grid_y + grid_h + 20
        for line in caption_lines:
            draw.text((grid_x, y), line, fill=(0, 0, 0), font=caption_font)
            bbox = draw.textbbox((0, 0), line, font=caption_font)
            y += (bbox[3] - bbox[1]) + 4

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    if args.save_metadata:
        meta_path = output_path.with_suffix(output_path.suffix + ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {meta_path}")
    log_to_wandb(args, output_path, metadata)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
