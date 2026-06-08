from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLOR_BY_CLASS = {
    0: (255, 255, 0),
    1: (0, 170, 255),
}
DEFAULT_NAMES = {
    0: "water",
    1: "muddy_water",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIMESTAMP_RE = re.compile(r"_(\d+)ms$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw YOLO segmentation polygon labels back onto dataset images."
    )
    parser.add_argument("--dataset", default="datasets/water_seg_full20")
    parser.add_argument("--output-dir", default="outputs/label_preview")
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--start-ms", type=float, default=0.0)
    parser.add_argument("--end-ms", type=float, default=0.0)
    parser.add_argument("--sample", choices=("ordered", "random"), default="ordered")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--hide-label-text", action="store_true")
    parser.add_argument(
        "--union-contours",
        action="store_true",
        help="Draw one merged contour per class instead of one outline per label polygon.",
    )
    parser.add_argument("--tile-width", type=int, default=520)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--contact-sheet", default="contact_sheet.jpg")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(dataset_dir / "data.yaml")
    pairs = discover_pairs(dataset_dir, args.split)
    pairs = filter_by_timestamp(pairs, args.start_ms, args.end_ms)
    pairs = select_pairs(pairs, args.max_images, args.sample, args.seed)
    if not pairs:
        raise SystemExit("no image/label pairs matched the requested filters")

    rendered = []
    individual_dir = output_dir / "images"
    if not args.no_individual:
        individual_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        canvas = render_overlay(
            image_path=image_path,
            label_path=label_path,
            names=names,
            alpha=args.alpha,
            line_thickness=args.line_thickness,
            hide_label_text=args.hide_label_text,
            union_contours=args.union_contours,
        )
        rendered.append((image_path, canvas))
        if not args.no_individual:
            out_path = individual_dir / f"{image_path.stem}_labels.jpg"
            cv2.imwrite(str(out_path), canvas)

    sheet = make_contact_sheet(rendered, names, args.tile_width, args.cols)
    sheet_path = output_dir / args.contact_sheet
    cv2.imwrite(str(sheet_path), sheet)
    print(f"pairs: {len(pairs)}")
    print(f"contact_sheet: {sheet_path.resolve()}")
    if not args.no_individual:
        print(f"individual_images: {individual_dir.resolve()}")
    return 0


def load_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        return DEFAULT_NAMES.copy()
    names: dict[int, str] = {}
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(\d+):\s*(.+)$", line)
        if not match:
            continue
        names[int(match.group(1))] = match.group(2).strip().strip("'\"")
    return names or DEFAULT_NAMES.copy()


def discover_pairs(dataset_dir: Path, split: str) -> list[tuple[Path, Path]]:
    splits = ("train", "val") if split == "all" else (split,)
    pairs: list[tuple[Path, Path]] = []
    for split_name in splits:
        image_dir = dataset_dir / "images" / split_name
        label_dir = dataset_dir / "labels" / split_name
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
    if pairs or split != "all":
        return pairs

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return pairs
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def filter_by_timestamp(
    pairs: list[tuple[Path, Path]],
    start_ms: float,
    end_ms: float,
) -> list[tuple[Path, Path]]:
    if start_ms <= 0 and end_ms <= 0:
        return pairs
    filtered = []
    for image_path, label_path in pairs:
        timestamp = timestamp_from_stem(image_path.stem)
        if timestamp is None:
            continue
        if start_ms > 0 and timestamp < start_ms:
            continue
        if end_ms > 0 and timestamp > end_ms:
            continue
        filtered.append((image_path, label_path))
    return filtered


def timestamp_from_stem(stem: str) -> int | None:
    match = TIMESTAMP_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


def select_pairs(
    pairs: list[tuple[Path, Path]],
    max_images: int,
    sample: str,
    seed: int,
) -> list[tuple[Path, Path]]:
    if max_images <= 0 or len(pairs) <= max_images:
        return pairs
    if sample == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(pairs, max_images))
    return pairs[:max_images]


def render_overlay(
    image_path: Path,
    label_path: Path,
    names: dict[int, str],
    alpha: float,
    line_thickness: int,
    hide_label_text: bool,
    union_contours: bool,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]
    overlay = image.copy()
    label_rows = read_label_rows(label_path, width, height)

    class_masks: dict[int, np.ndarray] = {}
    for class_id, points in label_rows:
        color = COLOR_BY_CLASS.get(class_id, color_for_class(class_id))
        cv2.fillPoly(overlay, [points], color)
        if union_contours:
            if class_id not in class_masks:
                class_masks[class_id] = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(class_masks[class_id], [points], 255)

    alpha = max(0.0, min(1.0, alpha))
    output = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    if union_contours:
        for class_id, mask in class_masks.items():
            color = COLOR_BY_CLASS.get(class_id, color_for_class(class_id))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.polylines(output, contours, True, color, max(1, line_thickness), lineType=cv2.LINE_AA)
    else:
        for class_id, points in label_rows:
            color = COLOR_BY_CLASS.get(class_id, color_for_class(class_id))
            cv2.polylines(output, [points], True, color, max(1, line_thickness), lineType=cv2.LINE_AA)
            if not hide_label_text:
                label = names.get(class_id, str(class_id))
                draw_label(output, label, points, color)

    draw_header(output, image_path, label_rows, names)
    return output


def read_label_rows(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    rows: list[tuple[int, np.ndarray]] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts[1:]) % 2:
            print(f"skip malformed label {label_path}:{line_no}", file=sys.stderr)
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"skip non-numeric label {label_path}:{line_no}", file=sys.stderr)
            continue
        points = []
        for x_norm, y_norm in zip(coords[::2], coords[1::2]):
            x = int(round(max(0.0, min(1.0, x_norm)) * (width - 1)))
            y = int(round(max(0.0, min(1.0, y_norm)) * (height - 1)))
            points.append((x, y))
        if len(points) >= 3:
            rows.append((class_id, np.array(points, dtype=np.int32)))
    return rows


def color_for_class(class_id: int) -> tuple[int, int, int]:
    hue = (class_id * 47) % 180
    hsv = np.array([[[hue, 190, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_label(
    image: np.ndarray,
    text: str,
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    x, y, _, _ = cv2.boundingRect(points)
    y = max(18, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x2 = min(image.shape[1] - 1, x + text_width + pad * 2)
    y1 = max(0, y - text_height - baseline - pad * 2)
    cv2.rectangle(image, (x, y1), (x2, y), color, -1)
    cv2.putText(
        image,
        text,
        (x + pad, y - baseline - pad),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_header(
    image: np.ndarray,
    image_path: Path,
    label_rows: list[tuple[int, np.ndarray]],
    names: dict[int, str],
) -> None:
    counts: dict[int, int] = {}
    for class_id, _ in label_rows:
        counts[class_id] = counts.get(class_id, 0) + 1
    count_text = ", ".join(f"{names.get(class_id, class_id)}:{count}" for class_id, count in sorted(counts.items()))
    text = f"{image_path.name} | {count_text or 'no labels'}"
    draw_banner(image, text)


def draw_banner(image: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    height = text_height + baseline + 12
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (20, 20, 20), -1)
    cv2.putText(
        image,
        text[:160],
        (8, height - baseline - 5),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def make_contact_sheet(
    rendered: list[tuple[Path, np.ndarray]],
    names: dict[int, str],
    tile_width: int,
    cols: int,
) -> np.ndarray:
    cols = max(1, cols)
    tile_width = max(160, tile_width)
    tiles = [resize_to_width(canvas, tile_width) for _, canvas in rendered]
    tile_height = max(tile.shape[0] for tile in tiles)
    rows = math.ceil(len(tiles) / cols)
    legend_height = 38
    sheet = np.full((rows * tile_height + legend_height, cols * tile_width, 3), 245, dtype=np.uint8)
    draw_legend(sheet[:legend_height], names)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        y = legend_height + row * tile_height
        x = col * tile_width
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return sheet


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    ratio = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * ratio)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_legend(image: np.ndarray, names: dict[int, str]) -> None:
    image[:] = (32, 32, 32)
    x = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "YOLO segmentation label preview", (x, 24), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    x += 300
    for class_id, name in sorted(names.items()):
        color = COLOR_BY_CLASS.get(class_id, color_for_class(class_id))
        cv2.rectangle(image, (x, 10), (x + 18, 28), color, -1)
        cv2.putText(image, name, (x + 26, 24), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        x += 26 + max(80, len(name) * 9)


if __name__ == "__main__":
    raise SystemExit(main())
