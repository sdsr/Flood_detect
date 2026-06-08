from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine flat YOLO segmentation datasets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        assert_safe_clean_target(output_dir)
        shutil.rmtree(output_dir)
    image_out = output_dir / "images"
    label_out = output_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for raw_dataset in args.dataset:
        dataset_dir = Path(raw_dataset)
        image_dir = dataset_dir / "images"
        label_dir = dataset_dir / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise SystemExit(f"dataset must contain flat images/labels dirs: {dataset_dir}")

        prefix = dataset_dir.name
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                skipped += 1
                continue
            stem = f"{prefix}__{image_path.stem}"
            shutil.copy2(image_path, image_out / f"{stem}{image_path.suffix.lower()}")
            shutil.copy2(label_path, label_out / f"{stem}.txt")
            copied += 1

    write_data_yaml(output_dir)
    print(f"copied: {copied}")
    print(f"skipped_without_label: {skipped}")
    print(f"output_dir: {output_dir.resolve()}")
    return 0


def assert_safe_clean_target(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(f"refusing to clean path outside workspace: {resolved}")


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


if __name__ == "__main__":
    raise SystemExit(main())
