from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.water_edge_segmenter import (  # noqa: E402
    parse_polygons,
    surface_preset_polygons,
)
from tools.build_pseudo_yolo_dataset import mask_to_yolo_lines  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create muddy-water YOLO segmentation labels from a seed label's "
            "muddy color distribution and fixed candidate polygons."
        )
    )
    parser.add_argument("--source", required=True, help="Video path")
    parser.add_argument("--output-dir", default="datasets/hakpa_harbor_muddy_seed20")
    parser.add_argument("--seed-image", required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--start-ms", type=float, default=0.0)
    parser.add_argument("--end-ms", type=float, default=0.0)
    parser.add_argument("--every-sec", type=float, default=6.0)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--morph-open", type=int, default=3)
    parser.add_argument("--morph-close", type=int, default=9)
    parser.add_argument("--seed-erode", type=int, default=3)
    parser.add_argument("--candidate-expand", type=int, default=8)
    parser.add_argument("--h-margin", type=float, default=8.0)
    parser.add_argument("--s-margin", type=float, default=18.0)
    parser.add_argument("--v-margin", type=float, default=24.0)
    parser.add_argument("--h-min", type=float, default=-1.0)
    parser.add_argument("--h-max", type=float, default=-1.0)
    parser.add_argument("--s-min", type=float, default=-1.0)
    parser.add_argument("--s-max", type=float, default=-1.0)
    parser.add_argument("--v-min", type=float, default=-1.0)
    parser.add_argument("--v-max", type=float, default=-1.0)
    parser.add_argument(
        "--surface-preset",
        default="none",
        choices=(
            "none",
            "yeongildae",
            "yeongildae-road",
            "yeongildae-road-strict",
            "hakpa-harbor-road",
        ),
    )
    parser.add_argument(
        "--surface-polygon",
        default=None,
        help="Normalized polygon points, e.g. '0,0.4 1,0.2 1,1 0,1'. Use ';' for multiple polygons.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        assert_safe_clean_target(output_dir)
        shutil.rmtree(output_dir)
    ensure_dataset_dirs(output_dir)
    write_data_yaml(output_dir)

    seed_image = cv2.imread(str(Path(args.seed_image)))
    if seed_image is None:
        raise SystemExit(f"failed to read seed image: {args.seed_image}")

    seed_label = Path(args.seed_label)
    seed_polygons = read_yolo_polygons(seed_label)
    if not seed_polygons:
        raise SystemExit(f"seed label has no polygons: {seed_label}")

    seed_mask = polygons_to_mask(seed_polygons, seed_image.shape[:2])
    if args.seed_erode > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            make_odd_kernel(args.seed_erode),
        )
        seed_mask = cv2.erode(seed_mask, kernel)

    thresholds = estimate_hsv_thresholds(seed_image, seed_mask, args)
    print(
        "hsv_thresholds:",
        f"H={thresholds[0]:.1f}-{thresholds[1]:.1f}",
        f"S={thresholds[2]:.1f}-{thresholds[3]:.1f}",
        f"V={thresholds[4]:.1f}-{thresholds[5]:.1f}",
    )

    surface_polygons = merge_polygons(
        surface_preset_polygons(args.surface_preset),
        parse_polygons(args.surface_polygon),
    )

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"failed to open source: {args.source}")

    saved = 0
    seek_ms = max(0.0, args.start_ms)
    step_ms = max(1.0, args.every_sec * 1000.0)
    while saved < args.max_images:
        if args.end_ms > 0 and seek_ms > args.end_ms:
            break
        frame, pos_ms = read_seek_frame(cap, seek_ms)
        seek_ms += step_ms
        if frame is None or pos_ms is None:
            break

        candidate_mask = polygons_to_mask(seed_polygons, frame.shape[:2])
        if args.candidate_expand > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                make_odd_kernel(args.candidate_expand),
            )
            candidate_mask = cv2.dilate(candidate_mask, kernel)
        if surface_polygons:
            surface_mask = polygons_to_mask(surface_polygons, frame.shape[:2])
            candidate_mask = cv2.bitwise_and(candidate_mask, surface_mask)

        muddy_mask = threshold_muddy(frame, candidate_mask, thresholds)
        muddy_mask = cleanup_mask(
            muddy_mask,
            min_area=args.min_area,
            open_kernel=args.morph_open,
            close_kernel=args.morph_close,
        )
        label_lines = mask_to_yolo_lines(muddy_mask, class_id=args.class_id, min_area=args.min_area)

        stem = f"muddy_seed_{saved:05d}_{int(pos_ms):08d}ms"
        cv2.imwrite(str(output_dir / "images" / f"{stem}.jpg"), frame)
        (output_dir / "labels" / f"{stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        saved += 1

    cap.release()
    print(f"saved {saved} images under {output_dir.resolve()}")
    return 0


def assert_safe_clean_target(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(f"refusing to clean path outside workspace: {resolved}")


def ensure_dataset_dirs(output_dir: Path) -> None:
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images",
                "val: images",
                "",
                "names:",
                "  0: water",
                "  1: muddy_water",
                "",
            ]
        ),
        encoding="utf-8",
    )


def read_yolo_polygons(label_path: Path) -> tuple[tuple[tuple[float, float], ...], ...]:
    polygons: list[tuple[tuple[float, float], ...]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(item) for item in parts[1:]]
        if len(coords) % 2 != 0:
            continue
        points = tuple((coords[i], coords[i + 1]) for i in range(0, len(coords), 2))
        if len(points) >= 3:
            polygons.append(points)
    return tuple(polygons)


def merge_polygons(*items) -> tuple[tuple[tuple[float, float], ...], ...]:
    merged: list[tuple[tuple[float, float], ...]] = []
    for item in items:
        if item:
            merged.extend(item)
    return tuple(merged)


def polygons_to_mask(
    polygons: tuple[tuple[tuple[float, float], ...], ...],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(
            [
                (
                    int(round(max(0.0, min(1.0, x)) * (width - 1))),
                    int(round(max(0.0, min(1.0, y)) * (height - 1))),
                )
                for x, y in polygon
            ],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def estimate_hsv_thresholds(image: np.ndarray, mask: np.ndarray, args) -> tuple[float, float, float, float, float, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        raise SystemExit("seed mask contains no pixels")

    broad = pixels[
        (pixels[:, 0] >= 5)
        & (pixels[:, 0] <= 60)
        & (pixels[:, 1] >= 8)
        & (pixels[:, 1] <= 190)
        & (pixels[:, 2] >= 40)
        & (pixels[:, 2] <= 240)
    ]
    if len(broad) < 100:
        broad = pixels

    h_low, h_high = np.percentile(broad[:, 0], [2, 98])
    s_low, s_high = np.percentile(broad[:, 1], [2, 98])
    v_low, v_high = np.percentile(broad[:, 2], [2, 98])

    h_low = apply_override(h_low - args.h_margin, args.h_min, lower=0, upper=179)
    h_high = apply_override(h_high + args.h_margin, args.h_max, lower=0, upper=179)
    s_low = apply_override(s_low - args.s_margin, args.s_min, lower=0, upper=255)
    s_high = apply_override(s_high + args.s_margin, args.s_max, lower=0, upper=255)
    v_low = apply_override(v_low - args.v_margin, args.v_min, lower=0, upper=255)
    v_high = apply_override(v_high + args.v_margin, args.v_max, lower=0, upper=255)
    return h_low, h_high, s_low, s_high, v_low, v_high


def apply_override(value: float, override: float, lower: float, upper: float) -> float:
    if override >= 0:
        value = override
    return float(max(lower, min(upper, value)))


def threshold_muddy(
    frame: np.ndarray,
    candidate_mask: np.ndarray,
    thresholds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    h_low, h_high, s_low, s_high, v_low, v_high = thresholds
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(mask, candidate_mask)


def cleanup_mask(mask: np.ndarray, min_area: int, open_kernel: int, close_kernel: int) -> np.ndarray:
    if open_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, make_odd_kernel(open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, make_odd_kernel(close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    labels, stats_mask = cv2.connectedComponentsWithStats(mask, 8)[1:3]
    cleaned = np.zeros_like(mask)
    for label_id in range(1, stats_mask.shape[0]):
        area = int(stats_mask[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def make_odd_kernel(size: int) -> tuple[int, int]:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return size, size


def read_seek_frame(cap: cv2.VideoCapture, target_ms: float):
    starts = [target_ms + offset for offset in (0, 250, 500, 1000, 2000, 4000)]
    for pos_ms in starts:
        if pos_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
        ok, frame = cap.read()
        if ok:
            return frame, pos_ms
    return None, None


if __name__ == "__main__":
    raise SystemExit(main())
