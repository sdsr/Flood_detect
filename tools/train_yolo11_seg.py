from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO11 segmentation model for water edges.")
    parser.add_argument("--data", default="datasets/water_seg/data.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained YOLO11 segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/water_seg")
    parser.add_argument("--name", default="yolo11n_water")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"dataset yaml not found: {data_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-yolo.txt"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        task="segment",
    )
    best = Path(getattr(model.trainer, "best", "")) if getattr(model, "trainer", None) is not None else Path()
    if not best:
        best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"best model: {best.resolve()}")
    print("copy it to: models/flood_seg/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
