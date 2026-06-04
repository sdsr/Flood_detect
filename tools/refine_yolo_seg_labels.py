from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_NAMES = {
    0: "water",
    1: "muddy_water",
}
CLASS_ALIASES = {
    "0": 0,
    "water": 0,
    "1": 1,
    "muddy": 1,
    "muddy_water": 1,
    "brown_water": 1,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIMESTAMP_RE = re.compile(r"_(\d+)ms$")


@dataclass(frozen=True)
class LabelPatch:
    name: str
    operation: str
    class_id: int
    polygon: tuple[tuple[float, float], ...]
    start_ms: float = 0.0
    end_ms: float = 0.0
    splits: tuple[str, ...] = ()
    stems: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a YOLO segmentation dataset while applying small mask cleanup "
            "and timestamp-limited polygon corrections."
        )
    )
    parser.add_argument("--dataset", default="datasets/water_seg_full20")
    parser.add_argument("--output-dir", default="datasets/water_seg_full20_refined")
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--patch-file", action="append", default=[])
    parser.add_argument(
        "--add-polygon",
        action="append",
        default=[],
        help=(
            "Add one correction: 'class:start_ms-end_ms:x,y x,y ...'. "
            "Example: muddy_water:1020000-1080000:0.2,0.4 0.5,0.3 0.6,0.4"
        ),
    )
    parser.add_argument(
        "--erase-polygon",
        action="append",
        default=[],
        help="Erase one correction with the same format as --add-polygon.",
    )
    parser.add_argument("--min-area", type=int, default=650)
    parser.add_argument("--open-kernel", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--fill-holes-classes", default="muddy_water")
    parser.add_argument("--smooth-ratio", type=float, default=0.0035)
    parser.add_argument(
        "--polygon-mode",
        choices=("contour", "grid-runs"),
        default="contour",
        help="Output polygon style. Use grid-runs to preserve holes from existing grid-run labels.",
    )
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--min-run-cells", type=int, default=3)
    parser.add_argument("--sample", choices=("ordered", "random"), default="ordered")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"dataset not found: {dataset_dir}")
    if args.clean and output_dir.exists():
        assert_safe_clean_target(output_dir)
        shutil.rmtree(output_dir)

    names = load_names(dataset_dir / "data.yaml")
    patches = load_patches(args.patch_file, names)
    patches.extend(parse_inline_patches(args.add_polygon, "add", names))
    patches.extend(parse_inline_patches(args.erase_polygon, "erase", names))
    fill_holes_classes = resolve_class_set(args.fill_holes_classes, names)

    pairs = discover_pairs(dataset_dir, args.split)
    pairs = select_pairs(pairs, args.max_images, args.sample, args.seed)
    if not pairs:
        raise SystemExit("no image/label pairs found")

    ensure_dataset_dirs(output_dir)
    write_data_yaml(output_dir, names)

    changed_count = 0
    patched_count = 0
    for split_name, image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}", file=sys.stderr)
            continue
        masks = labels_to_masks(label_path, image.shape[:2], names)
        before_labels = masks_to_label_lines(
            masks,
            min_area=args.min_area,
            smooth_ratio=args.smooth_ratio,
            polygon_mode=args.polygon_mode,
            grid_size=args.grid_size,
            min_run_cells=args.min_run_cells,
        )

        timestamp_ms = timestamp_from_stem(image_path.stem)
        matched_patches = [
            patch
            for patch in patches
            if patch_matches(patch, split_name, image_path.stem, timestamp_ms)
        ]
        for patch in matched_patches:
            apply_patch_to_masks(masks, patch, image.shape[:2])
        if matched_patches:
            patched_count += 1

        masks = cleanup_masks(
            masks,
            min_area=args.min_area,
            open_kernel=args.open_kernel,
            close_kernel=args.close_kernel,
            fill_holes_classes=fill_holes_classes,
        )
        prioritize_muddy(masks)
        after_labels = masks_to_label_lines(
            masks,
            min_area=args.min_area,
            smooth_ratio=args.smooth_ratio,
            polygon_mode=args.polygon_mode,
            grid_size=args.grid_size,
            min_run_cells=args.min_run_cells,
        )
        if before_labels != after_labels:
            changed_count += 1

        out_image = output_dir / "images" / split_name / image_path.name
        out_label = output_dir / "labels" / split_name / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(after_labels) + ("\n" if after_labels else ""), encoding="utf-8")

    print(f"pairs: {len(pairs)}")
    print(f"patched_images: {patched_count}")
    print(f"changed_labels: {changed_count}")
    print(f"output_dir: {output_dir.resolve()}")
    return 0


def assert_safe_clean_target(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(f"refusing to clean path outside workspace: {resolved}")


def load_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        return DEFAULT_NAMES.copy()
    names: dict[int, str] = {}
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(\d+):\s*(.+)$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip().strip("'\"")
    return names or DEFAULT_NAMES.copy()


def discover_pairs(dataset_dir: Path, split: str) -> list[tuple[str, Path, Path]]:
    split_names = ("train", "val") if split == "all" else (split,)
    pairs: list[tuple[str, Path, Path]] = []
    for split_name in split_names:
        image_dir = dataset_dir / "images" / split_name
        label_dir = dataset_dir / "labels" / split_name
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            pairs.append((split_name, image_path, label_path))
    return pairs


def select_pairs(
    pairs: list[tuple[str, Path, Path]],
    max_images: int,
    sample: str,
    seed: int,
) -> list[tuple[str, Path, Path]]:
    if max_images <= 0 or len(pairs) <= max_images:
        return pairs
    if sample == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(pairs, max_images), key=lambda item: str(item[1]))
    return pairs[:max_images]


def ensure_dataset_dirs(output_dir: Path) -> None:
    for child in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path, names: dict[int, str]) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for class_id, name in sorted(names.items()):
        lines.append(f"  {class_id}: {name}")
    lines.append("")
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def labels_to_masks(
    label_path: Path,
    shape: tuple[int, int],
    names: dict[int, str],
) -> dict[int, np.ndarray]:
    height, width = shape
    masks = {class_id: np.zeros((height, width), dtype=np.uint8) for class_id in names}
    if not label_path.exists():
        return masks
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
        if class_id not in masks:
            masks[class_id] = np.zeros((height, width), dtype=np.uint8)
        points = []
        for x_norm, y_norm in zip(coords[::2], coords[1::2]):
            x = int(round(max(0.0, min(1.0, x_norm)) * (width - 1)))
            y = int(round(max(0.0, min(1.0, y_norm)) * (height - 1)))
            points.append((x, y))
        if len(points) >= 3:
            cv2.fillPoly(masks[class_id], [np.array(points, dtype=np.int32)], 255)
    return masks


def cleanup_masks(
    masks: dict[int, np.ndarray],
    min_area: int,
    open_kernel: int,
    close_kernel: int,
    fill_holes_classes: set[int],
) -> dict[int, np.ndarray]:
    cleaned: dict[int, np.ndarray] = {}
    for class_id, mask in masks.items():
        result = mask.copy()
        result = morph(result, cv2.MORPH_OPEN, open_kernel)
        result = morph(result, cv2.MORPH_CLOSE, close_kernel)
        if class_id in fill_holes_classes:
            result = fill_external_regions(result)
        result = filter_small_components(result, min_area)
        cleaned[class_id] = result
    return cleaned


def morph(mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, operation, kernel)


def fill_external_regions(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.fillPoly(filled, contours, 255)
    return filled


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def prioritize_muddy(masks: dict[int, np.ndarray]) -> None:
    water_id = resolve_class_id("water", masks.keys())
    muddy_id = resolve_class_id("muddy_water", masks.keys())
    if water_id is None or muddy_id is None:
        return
    masks[water_id][masks[muddy_id] > 0] = 0


def masks_to_label_lines(
    masks: dict[int, np.ndarray],
    min_area: int,
    smooth_ratio: float,
    polygon_mode: str,
    grid_size: int,
    min_run_cells: int,
) -> list[str]:
    lines: list[str] = []
    for class_id, mask in sorted(masks.items()):
        if polygon_mode == "grid-runs":
            lines.extend(
                mask_to_grid_run_lines(
                    mask,
                    class_id=class_id,
                    grid_size=grid_size,
                    min_run_cells=min_run_cells,
                )
            )
            continue
        for contour in contours_from_mask(mask, min_area, smooth_ratio):
            points = contour.reshape(-1, 2)
            if len(points) < 3:
                continue
            height, width = mask.shape[:2]
            normalized: list[str] = []
            for x, y in points:
                normalized.append(f"{max(0.0, min(1.0, x / width)):.6f}")
                normalized.append(f"{max(0.0, min(1.0, y / height)):.6f}")
            if len(normalized) >= 6:
                lines.append(f"{class_id} " + " ".join(normalized))
    return lines


def mask_to_grid_run_lines(
    mask: np.ndarray,
    class_id: int,
    grid_size: int,
    min_run_cells: int,
) -> list[str]:
    height, width = mask.shape[:2]
    grid_size = max(2, int(grid_size))
    min_run_cells = max(1, int(min_run_cells))
    small_width = max(1, (width + grid_size - 1) // grid_size)
    small_height = max(1, (height + grid_size - 1) // grid_size)
    small = cv2.resize(mask, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small = small >= 128

    lines: list[str] = []
    for y_cell in range(small_height):
        x_cell = 0
        while x_cell < small_width:
            while x_cell < small_width and not small[y_cell, x_cell]:
                x_cell += 1
            start = x_cell
            while x_cell < small_width and small[y_cell, x_cell]:
                x_cell += 1
            end = x_cell
            if end - start < min_run_cells:
                continue
            x0 = start * grid_size
            x1 = min(width, end * grid_size)
            y0 = y_cell * grid_size
            y1 = min(height, (y_cell + 1) * grid_size)
            if x1 <= x0 or y1 <= y0:
                continue
            points = (
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            )
            normalized = []
            for x, y in points:
                normalized.append(f"{max(0.0, min(1.0, x / width)):.6f}")
                normalized.append(f"{max(0.0, min(1.0, y / height)):.6f}")
            lines.append(f"{class_id} " + " ".join(normalized))
    return lines


def contours_from_mask(mask: np.ndarray, min_area: int, smooth_ratio: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(1.0, max(0.0, smooth_ratio) * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            kept.append(approx)
    return kept


def load_patches(paths: Iterable[str], names: dict[int, str]) -> list[LabelPatch]:
    patches: list[LabelPatch] = []
    for raw_path in paths:
        path = Path(raw_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_patches = data.get("patches", data if isinstance(data, list) else [])
        for idx, item in enumerate(raw_patches):
            patches.append(patch_from_mapping(item, names, f"{path.name}:{idx}"))
    return patches


def patch_from_mapping(item: dict, names: dict[int, str], fallback_name: str) -> LabelPatch:
    operation = str(item.get("operation", "add")).lower().strip()
    if operation not in {"add", "erase"}:
        raise ValueError(f"patch operation must be add or erase: {fallback_name}")
    class_id = resolve_class_id(str(item.get("class", "muddy_water")), names)
    if class_id is None:
        raise ValueError(f"unknown patch class in {fallback_name}: {item.get('class')}")
    polygon = parse_polygon_points(item.get("polygon", []))
    return LabelPatch(
        name=str(item.get("name", fallback_name)),
        operation=operation,
        class_id=class_id,
        polygon=polygon,
        start_ms=float(item.get("start_ms", 0.0) or 0.0),
        end_ms=float(item.get("end_ms", 0.0) or 0.0),
        splits=tuple(str(value) for value in item.get("splits", []) or ()),
        stems=tuple(str(value) for value in item.get("stems", []) or ()),
    )


def parse_inline_patches(
    values: Iterable[str],
    operation: str,
    names: dict[int, str],
) -> list[LabelPatch]:
    patches: list[LabelPatch] = []
    for index, value in enumerate(values):
        try:
            class_part, range_part, points_part = value.split(":", 2)
        except ValueError as exc:
            raise ValueError(f"invalid polygon patch: {value}") from exc
        class_id = resolve_class_id(class_part, names)
        if class_id is None:
            raise ValueError(f"unknown patch class: {class_part}")
        start_ms, end_ms = parse_time_range(range_part)
        patches.append(
            LabelPatch(
                name=f"inline_{operation}_{index}",
                operation=operation,
                class_id=class_id,
                polygon=parse_polygon_text(points_part),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return patches


def parse_time_range(value: str) -> tuple[float, float]:
    if "-" not in value:
        timestamp = float(value)
        return timestamp, timestamp
    start, end = value.split("-", 1)
    return float(start or 0.0), float(end or 0.0)


def parse_polygon_points(value: object) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    if isinstance(value, str):
        return parse_polygon_text(value)
    if not isinstance(value, list):
        raise ValueError("patch polygon must be a list of [x, y] points or a point string")
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("patch polygon point must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("patch polygon coordinates must be normalized from 0 to 1")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("patch polygon must contain at least three points")
    return tuple(points)


def parse_polygon_text(value: str) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for raw_point in value.split():
        x_raw, y_raw = raw_point.split(",", 1)
        x, y = float(x_raw), float(y_raw)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("patch polygon coordinates must be normalized from 0 to 1")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("patch polygon must contain at least three points")
    return tuple(points)


def resolve_class_set(value: str, names: dict[int, str]) -> set[int]:
    if not value.strip() or value.strip().lower() in {"none", "off", "false"}:
        return set()
    class_ids = set()
    for item in value.split(","):
        class_id = resolve_class_id(item.strip(), names)
        if class_id is None:
            raise ValueError(f"unknown class: {item}")
        class_ids.add(class_id)
    return class_ids


def resolve_class_id(value: str, names: dict[int, str] | Iterable[int]) -> int | None:
    normalized = str(value).strip().lower()
    if isinstance(names, dict):
        for class_id, name in names.items():
            if normalized in {str(class_id), str(name).lower()}:
                return class_id
    else:
        for class_id in names:
            if normalized == str(class_id):
                return class_id
    return CLASS_ALIASES.get(normalized)


def patch_matches(
    patch: LabelPatch,
    split_name: str,
    stem: str,
    timestamp_ms: int | None,
) -> bool:
    if patch.splits and split_name not in patch.splits:
        return False
    if patch.stems and stem not in patch.stems:
        return False
    if patch.start_ms > 0 or patch.end_ms > 0:
        if timestamp_ms is None:
            return False
        if patch.start_ms > 0 and timestamp_ms < patch.start_ms:
            return False
        if patch.end_ms > 0 and timestamp_ms > patch.end_ms:
            return False
    return True


def apply_patch_to_masks(
    masks: dict[int, np.ndarray],
    patch: LabelPatch,
    shape: tuple[int, int],
) -> None:
    height, width = shape
    if patch.class_id not in masks:
        masks[patch.class_id] = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [
            [
                int(round(max(0.0, min(1.0, x)) * (width - 1))),
                int(round(max(0.0, min(1.0, y)) * (height - 1))),
            ]
            for x, y in patch.polygon
        ],
        dtype=np.int32,
    )
    if patch.operation == "add":
        cv2.fillPoly(masks[patch.class_id], [polygon], 255)
    else:
        cv2.fillPoly(masks[patch.class_id], [polygon], 0)


def timestamp_from_stem(stem: str) -> int | None:
    match = TIMESTAMP_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
