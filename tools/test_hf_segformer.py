from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Hugging Face SegFormer semantic-segmentation mask on one image."
    )
    parser.add_argument("--model", required=True, help="HF model id, e.g. rbh227/floodnet-segformer")
    parser.add_argument(
        "--processor-model",
        default=None,
        help="Optional HF model id to load the image processor from when the model repo has no preprocessor_config.json",
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output preview image path")
    parser.add_argument(
        "--classes",
        default="water,road flooded",
        help="Comma-separated class names to keep, matched case-insensitively",
    )
    parser.add_argument("--alpha", type=float, default=0.32)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--color", default="0,170,255", help="B,G,R overlay/edge color")
    return parser.parse_args()


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--color must be B,G,R")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)  # type: ignore[return-value]


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image = Image.open(image_path).convert("RGB")
    processor_source = args.processor_model or args.model
    processor = AutoImageProcessor.from_pretrained(processor_source)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=(pil_image.height, pil_image.width),
        mode="bilinear",
        align_corners=False,
    )
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    requested = {item.strip().lower() for item in args.classes.split(",") if item.strip()}
    keep_ids = {idx for idx, label in id2label.items() if label.lower() in requested}
    if not keep_ids:
        raise SystemExit(f"no matching classes in model labels: {id2label}")

    mask = np.isin(pred, list(keep_ids)).astype(np.uint8) * 255
    if args.min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= args.min_area:
                filtered[labels == i] = 255
        mask = filtered

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    color = parse_color(args.color)
    overlay = frame.copy()
    overlay[mask > 0] = (
        overlay[mask > 0].astype(np.float32) * (1.0 - args.alpha)
        + np.array(color, dtype=np.float32) * args.alpha
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= args.min_area]
    cv2.polylines(overlay, contours, True, color, 3, lineType=cv2.LINE_AA)
    cv2.imwrite(str(output_path), overlay)

    counts = {
        id2label[idx]: int((pred == idx).sum())
        for idx in sorted(id2label)
    }
    summary = {
        "model": args.model,
        "image": str(image_path),
        "output": str(output_path),
        "labels": id2label,
        "kept_classes": [id2label[idx] for idx in sorted(keep_ids)],
        "mask_pixels_after_filter": int((mask > 0).sum()),
        "class_pixel_counts": counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
