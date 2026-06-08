from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.water_edge_segmenter import EdgeConfig, WaterEdgeSegmenter, contours_from_mask  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a flat YOLO segmentation label-editing dataset from an image folder "
            "using a YOLO segmentation model as the first-pass annotator."
        )
    )
    parser.add_argument("--source-images", required=True, help="Folder containing extracted frames.")
    parser.add_argument("--output-dir", required=True, help="Flat dataset folder to create.")
    parser.add_argument("--model", required=True, help="YOLO segmentation checkpoint.")
    parser.add_argument("--backend", default="yolo11", choices=("yolo11", "yolo26", "yolo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=1500)
    parser.add_argument("--morph-kernel", type=int, default=7)
    parser.add_argument("--edge-smooth-ratio", type=float, default=0.0015)
    parser.add_argument("--edge-curve-iterations", type=int, default=0)
    parser.add_argument("--class-id", type=int, default=1, help="YOLO class id to write for bootstrapped masks.")
    parser.add_argument(
        "--mode",
        choices=("combined-as-muddy", "layers"),
        default="combined-as-muddy",
        help="combined-as-muddy writes all model masks as class-id; layers preserves water=0 and muddy=1.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place images in the output dataset.",
    )
    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        help="Overwrite labels in an existing output dataset. Images are kept/copied as needed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_dir = Path(args.source_images)
    output_dir = Path(args.output_dir)
    image_out = output_dir / "images"
    label_out = output_dir / "labels"

    if not source_dir.exists():
        raise SystemExit(f"source image dir not found: {source_dir}")
    images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found in: {source_dir}")
    if output_dir.exists() and any((output_dir / child).exists() for child in ("images", "labels")):
        if not args.overwrite_labels:
            raise SystemExit(
                f"output dataset already exists: {output_dir}\n"
                "Use --overwrite-labels only when you intentionally want to replace existing label files."
            )

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    write_data_yaml(output_dir)

    segmenter = WaterEdgeSegmenter(
        EdgeConfig(
            backend=args.backend,
            yolo_device=args.device,
            yolo_imgsz=args.imgsz,
            yolo_conf=args.conf,
            min_area=args.min_area,
            morph_kernel=args.morph_kernel,
            edge_smooth_ratio=args.edge_smooth_ratio,
            edge_curve_iterations=args.edge_curve_iterations,
        ),
        model_path=args.model,
    )

    labeled = 0
    empty = 0
    polygons = 0
    for index, src_path in enumerate(images, start=1):
        dst_image = image_out / src_path.name
        if not dst_image.exists():
            place_image(src_path, dst_image, args.copy_mode)

        frame = cv2.imread(str(src_path))
        if frame is None:
            print(f"skip unreadable image: {src_path}")
            continue

        masks = segmenter.segment_layers(frame)
        if args.mode == "layers":
            lines = []
            lines.extend(mask_to_yolo_lines(masks.regular, 0, args.min_area, args.edge_smooth_ratio, args.edge_curve_iterations))
            lines.extend(mask_to_yolo_lines(masks.muddy, 1, args.min_area, args.edge_smooth_ratio, args.edge_curve_iterations))
        else:
            lines = mask_to_yolo_lines(
                masks.combined,
                args.class_id,
                args.min_area,
                args.edge_smooth_ratio,
                args.edge_curve_iterations,
            )

        label_path = label_out / f"{src_path.stem}.txt"
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        if lines:
            labeled += 1
            polygons += len(lines)
        else:
            empty += 1
        if index == 1 or index % 25 == 0 or index == len(images):
            print(f"{index}/{len(images)} images processed")

    print(f"dataset: {output_dir.resolve()}")
    print(f"images: {len(images)}")
    print(f"labeled_images: {labeled}")
    print(f"empty_images: {empty}")
    print(f"polygons: {polygons}")
    return 0


def place_image(src_path: Path, dst_path: Path, copy_mode: str) -> None:
    if copy_mode == "hardlink":
        try:
            dst_path.hardlink_to(src_path.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src_path, dst_path)


def write_data_yaml(output_dir: Path) -> None:
    path_text = str(output_dir.resolve()).replace("\\", "/")
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {path_text}",
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


def mask_to_yolo_lines(
    mask,
    class_id: int,
    min_area: int,
    smooth_ratio: float,
    curve_iterations: int,
) -> list[str]:
    height, width = mask.shape[:2]
    lines: list[str] = []
    for contour in contours_from_mask(mask, min_area, smooth_ratio, curve_iterations):
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        coords = []
        for x, y in points:
            coords.append(f"{max(0.0, min(1.0, float(x) / width)):.6f}")
            coords.append(f"{max(0.0, min(1.0, float(y) / height)):.6f}")
        if len(coords) >= 6:
            lines.append(f"{class_id} " + " ".join(coords))
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
