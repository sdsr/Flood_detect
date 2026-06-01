from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.water_edge_segmenter import (  # noqa: E402
    EdgeConfig,
    WaterEdgeSegmenter,
    contours_from_mask,
    parse_bgr,
    parse_roi,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a YOLO segmentation dataset from the current heuristic water masks."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="datasets/water_seg")
    parser.add_argument("--start-ms", type=float, default=1000)
    parser.add_argument("--end-ms", type=float, default=0)
    parser.add_argument("--every-sec", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clean", action="store_true")

    parser.add_argument("--roi", default=None, help="Normalized ROI as x0,y0,x1,y1. Omit for full-frame labels.")
    parser.add_argument("--min-area", type=int, default=6500)
    parser.add_argument("--morph-kernel", type=int, default=15)
    parser.add_argument("--border-margin", type=int, default=8)
    parser.add_argument("--sat-max", type=int, default=92)
    parser.add_argument("--value-percentile", type=float, default=57)
    parser.add_argument("--min-value", type=int, default=45)
    parser.add_argument("--max-value", type=int, default=222)
    parser.add_argument("--texture-std-max", type=float, default=22)
    parser.add_argument("--muddy-hue-min", type=int, default=8)
    parser.add_argument("--muddy-hue-max", type=int, default=45)
    parser.add_argument("--muddy-sat-min", type=int, default=18)
    parser.add_argument("--muddy-sat-max", type=int, default=185)
    parser.add_argument("--muddy-value-min", type=int, default=50)
    parser.add_argument("--muddy-value-max", type=int, default=225)
    parser.add_argument("--muddy-texture-std-max", type=float, default=36)
    parser.add_argument("--water-color", default="255,255,0")
    parser.add_argument("--muddy-color", default="0,170,255")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.clean:
        clean_dataset_dirs(output_dir)
    ensure_dataset_dirs(output_dir)
    write_data_yaml(output_dir)

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"failed to open source: {args.source}")

    seek_ms = max(0.0, args.start_ms)
    step_ms = max(1.0, args.every_sec * 1000.0)

    segmenter = WaterEdgeSegmenter(
        EdgeConfig(
            backend="heuristic",
            roi=parse_roi(args.roi),
            water_color=parse_bgr(args.water_color),
            muddy_color=parse_bgr(args.muddy_color),
            min_area=args.min_area,
            morph_kernel=args.morph_kernel,
            border_margin=args.border_margin,
            sat_max=args.sat_max,
            value_percentile=args.value_percentile,
            min_value=args.min_value,
            max_value=args.max_value,
            texture_std_max=args.texture_std_max,
            muddy_hue_min=args.muddy_hue_min,
            muddy_hue_max=args.muddy_hue_max,
            muddy_sat_min=args.muddy_sat_min,
            muddy_sat_max=args.muddy_sat_max,
            muddy_value_min=args.muddy_value_min,
            muddy_value_max=args.muddy_value_max,
            muddy_texture_std_max=args.muddy_texture_std_max,
        )
    )

    random.seed(args.seed)
    saved = 0
    while saved < args.max_images:
        frame, pos_ms = read_seek_frame(cap, seek_ms)
        seek_ms += step_ms
        if frame is None or pos_ms is None:
            break
        if args.end_ms > 0 and pos_ms > args.end_ms:
            break

        masks = segmenter.segment_layers(frame)
        label_lines = []
        label_lines.extend(mask_to_yolo_lines(masks.regular, class_id=0, min_area=args.min_area))
        label_lines.extend(mask_to_yolo_lines(masks.muddy, class_id=1, min_area=args.min_area))
        if not label_lines:
            continue

        split = "val" if random.random() < args.val_ratio else "train"
        stem = f"pseudo_{saved:05d}_{int(pos_ms):08d}ms"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(image_path), frame)
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        saved += 1

    cap.release()
    print(f"saved {saved} pseudo-labeled images under {output_dir.resolve()}")
    return 0


def clean_dataset_dirs(output_dir: Path) -> None:
    for child in ("images", "labels"):
        path = output_dir / child
        if path.exists():
            shutil.rmtree(path)


def ensure_dataset_dirs(output_dir: Path) -> None:
    for child in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: water",
                "  1: muddy_water",
                "",
            ]
        ),
        encoding="utf-8",
    )


def mask_to_yolo_lines(mask, class_id: int, min_area: int) -> list[str]:
    height, width = mask.shape[:2]
    lines: list[str] = []
    for contour in contours_from_mask(mask, min_area):
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        normalized = []
        for x, y in points:
            normalized.append(f"{max(0.0, min(1.0, x / width)):.6f}")
            normalized.append(f"{max(0.0, min(1.0, y / height)):.6f}")
        if len(normalized) >= 6:
            lines.append(f"{class_id} " + " ".join(normalized))
    return lines


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def read_seek_frame(cap: cv2.VideoCapture, start_ms: float):
    if start_ms > 0:
        starts = [start_ms + offset for offset in (0, 500, 1000, 2000, 4000, 8000)]
    else:
        starts = [0, 500, 1000, 2000, 4000, 8000]
    for pos_ms in starts:
        if pos_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
        ok, frame = cap.read()
        if ok:
            if pos_ms != start_ms:
                print(f"first readable frame: {pos_ms:.0f}ms")
            return frame, pos_ms
    return None, None


if __name__ == "__main__":
    raise SystemExit(main())
