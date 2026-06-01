from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


ColorBGR = tuple[int, int, int]
Roi = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class EdgeConfig:
    backend: str = "heuristic"
    roi: Optional[Roi] = None
    water_color: ColorBGR = (255, 255, 0)
    muddy_color: ColorBGR = (0, 170, 255)
    line_thickness: int = 3
    mask_alpha: float = 0.16
    suppress_roi_border_edges: bool = True
    min_area: int = 800
    morph_kernel: int = 9
    border_margin: int = 6
    sat_max: int = 95
    value_percentile: float = 58.0
    min_value: int = 45
    max_value: int = 235
    texture_std_max: float = 24.0
    muddy_water_enabled: bool = True
    muddy_hue_min: int = 8
    muddy_hue_max: int = 42
    muddy_sat_min: int = 22
    muddy_sat_max: int = 175
    muddy_value_min: int = 55
    muddy_value_max: int = 225
    muddy_texture_std_max: float = 34.0
    yolo_conf: float = 0.35
    yolo_imgsz: int = 640
    yolo_device: str = "cpu"
    yolo_classes: Optional[tuple[str, ...]] = None
    hybrid_muddy: bool = False
    surface_polygons: Optional[tuple[Polygon, ...]] = None
    suppress_surface_border_edges: bool = True
    roboflow_api_key: Optional[str] = None
    roboflow_endpoint: str = "river-flood-detection/5"
    roboflow_api_url: str = "https://serverless.roboflow.com"
    roboflow_confidence: float = 0.25


@dataclass(frozen=True)
class SegmentationMasks:
    regular: np.ndarray
    muddy: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        return cv2.bitwise_or(self.regular, self.muddy)


class WaterEdgeSegmenter:
    def __init__(self, config: EdgeConfig, model_path: Optional[str] = None) -> None:
        self.config = config
        self.model = None
        self.model_names: dict[int, str] = {}
        self._surface_mask_shape: Optional[tuple[int, int]] = None
        self._surface_mask: Optional[np.ndarray] = None
        backend = config.backend.lower().strip()
        if backend == "yolo11":
            backend = "yolo"
        if backend == "river":
            backend = "roboflow"
        if backend not in {"heuristic", "yolo", "roboflow"}:
            raise ValueError("backend must be 'heuristic', 'yolo', 'yolo11', 'roboflow', or 'river'")
        self.backend = backend
        if self.backend == "yolo":
            if not model_path:
                raise ValueError("--seg-model is required when --backend yolo is used")
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Segmentation model not found: {path}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO backend requires ultralytics. Install with: "
                    ".\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-yolo.txt"
                ) from exc
            self.model = YOLO(str(path))
            names = getattr(self.model, "names", {}) or {}
            self.model_names = {int(k): str(v).lower() for k, v in names.items()}
        if self.backend == "roboflow":
            if not (self.config.roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")):
                raise ValueError("Roboflow backend requires --roboflow-api-key or ROBOFLOW_API_KEY")
            try:
                import requests  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Roboflow backend requires requests. Install with: "
                    ".\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-yolo.txt"
                ) from exc

    def segment_layers(self, frame: np.ndarray) -> SegmentationMasks:
        x0, y0, x1, y1 = roi_to_pixels(frame.shape, self.config.roi)
        roi_frame = frame[y0:y1, x0:x1]
        if roi_frame.size == 0:
            empty = np.zeros(frame.shape[:2], dtype=np.uint8)
            return SegmentationMasks(regular=empty, muddy=empty)

        if self.backend == "yolo":
            regular_roi, muddy_roi = self._segment_yolo(roi_frame)
            if self.config.hybrid_muddy:
                _, heuristic_muddy = self._segment_heuristic(roi_frame)
                muddy_roi = cv2.bitwise_or(muddy_roi, heuristic_muddy)
        elif self.backend == "roboflow":
            regular_roi, muddy_roi = self._segment_roboflow(roi_frame)
        else:
            regular_roi, muddy_roi = self._segment_heuristic(roi_frame)

        surface_mask = self._surface_mask_for_shape(frame.shape)
        if surface_mask is not None:
            surface_roi_mask = surface_mask[y0:y1, x0:x1]
            regular_roi = cv2.bitwise_and(regular_roi, surface_roi_mask)
            muddy_roi = cv2.bitwise_and(muddy_roi, surface_roi_mask)

        regular_roi = postprocess_mask(
            regular_roi,
            min_area=self.config.min_area,
            kernel_size=self.config.morph_kernel,
            border_margin=self.config.border_margin,
        )
        muddy_roi = postprocess_mask(
            muddy_roi,
            min_area=self.config.min_area,
            kernel_size=self.config.morph_kernel,
            border_margin=self.config.border_margin,
        )
        # Muddy water gets priority where the two rules overlap, so the overlay
        # stays visually meaningful instead of collapsing back into one color.
        regular_roi[muddy_roi > 0] = 0

        regular = np.zeros(frame.shape[:2], dtype=np.uint8)
        muddy = np.zeros(frame.shape[:2], dtype=np.uint8)
        regular[y0:y1, x0:x1] = regular_roi
        muddy[y0:y1, x0:x1] = muddy_roi
        return SegmentationMasks(regular=regular, muddy=muddy)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        return self.segment_layers(frame).combined

    def draw(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        masks = self.segment_layers(frame)
        mask = masks.combined
        contours = contours_from_mask(mask, self.config.min_area)
        output = frame.copy()
        if self.config.mask_alpha > 0:
            overlay = output.copy()
            overlay[masks.regular > 0] = blend_color(
                overlay[masks.regular > 0],
                self.config.water_color,
                0.55,
            )
            overlay[masks.muddy > 0] = blend_color(
                overlay[masks.muddy > 0],
                self.config.muddy_color,
                0.55,
            )
            output = cv2.addWeighted(overlay, self.config.mask_alpha, output, 1 - self.config.mask_alpha, 0)
        self._draw_layer_edges(output, frame.shape[:2], masks.regular, self.config.water_color)
        self._draw_layer_edges(output, frame.shape[:2], masks.muddy, self.config.muddy_color)
        return output, mask, contours

    def _draw_layer_edges(
        self,
        output: np.ndarray,
        shape: tuple[int, int],
        mask: np.ndarray,
        color: ColorBGR,
    ) -> None:
        contours = contours_from_mask(mask, self.config.min_area)
        if not contours:
            return
        should_suppress_surface = (
            self.config.suppress_surface_border_edges
            and self.config.surface_polygons is not None
        )
        if (self.config.suppress_roi_border_edges and self.config.roi is not None) or should_suppress_surface:
            edge_mask = edge_mask_from_contours(shape, contours, self.config.line_thickness)
            margin = max(self.config.border_margin * 3, self.config.line_thickness * 4)
            if self.config.suppress_roi_border_edges and self.config.roi is not None:
                x0, y0, x1, y1 = roi_to_pixels(output.shape, self.config.roi)
                edge_mask[y0:y0 + margin, x0:x1] = 0
                edge_mask[y1 - margin:y1, x0:x1] = 0
                edge_mask[y0:y1, x0:x0 + margin] = 0
                edge_mask[y0:y1, x1 - margin:x1] = 0
            if should_suppress_surface:
                surface_mask = self._surface_mask_for_shape(shape)
                if surface_mask is not None:
                    edge_mask[surface_border_mask(surface_mask, margin) > 0] = 0
            output[edge_mask > 0] = color
            return
        cv2.polylines(
            output,
            contours,
            isClosed=True,
            color=color,
            thickness=self.config.line_thickness,
            lineType=cv2.LINE_AA,
        )

    def _segment_heuristic(self, roi_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blurred = cv2.bilateralFilter(roi_frame, 7, 45, 45)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        value_threshold = max(
            self.config.min_value,
            int(np.percentile(value, self.config.value_percentile)),
        )

        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (13, 13))
        mean_sq = cv2.blur(gray_f * gray_f, (13, 13))
        local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

        low_saturation = saturation <= self.config.sat_max
        reflective_or_bright = value >= value_threshold
        not_white_marking = value <= self.config.max_value
        smooth_surface = local_std <= self.config.texture_std_max
        reflective_water = low_saturation & reflective_or_bright & not_white_marking & smooth_surface

        if self.config.muddy_water_enabled:
            muddy_hue = (hue >= self.config.muddy_hue_min) & (hue <= self.config.muddy_hue_max)
            muddy_saturation = (
                (saturation >= self.config.muddy_sat_min)
                & (saturation <= self.config.muddy_sat_max)
            )
            muddy_value = (
                (value >= self.config.muddy_value_min)
                & (value <= self.config.muddy_value_max)
            )
            muddy_smooth = local_std <= self.config.muddy_texture_std_max
            muddy_water = muddy_hue & muddy_saturation & muddy_value & muddy_smooth
        else:
            muddy_water = np.zeros_like(reflective_water, dtype=bool)

        return reflective_water.astype(np.uint8) * 255, muddy_water.astype(np.uint8) * 255

    def _segment_yolo(self, roi_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        results = self.model.predict(
            source=roi_frame,
            imgsz=self.config.yolo_imgsz,
            conf=self.config.yolo_conf,
            device=self.config.yolo_device,
            verbose=False,
        )
        regular = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
        muddy = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
        if not results:
            return regular, muddy
        result = results[0]
        if getattr(result, "masks", None) is None or result.masks is None:
            return regular, muddy

        class_filter = normalize_class_filter(self.config.yolo_classes)
        boxes = getattr(result, "boxes", None)
        cls_values = []
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_values = [int(v) for v in boxes.cls.detach().cpu().numpy().tolist()]

        masks = result.masks.data.detach().cpu().numpy()
        for idx, raw_mask in enumerate(masks):
            if class_filter is not None:
                cls_id = cls_values[idx] if idx < len(cls_values) else None
                cls_name = self.model_names.get(cls_id, "") if cls_id is not None else ""
                if str(cls_id) not in class_filter and cls_name not in class_filter:
                    continue
            resized = cv2.resize(raw_mask, (roi_frame.shape[1], roi_frame.shape[0]))
            cls_id = cls_values[idx] if idx < len(cls_values) else None
            cls_name = self.model_names.get(cls_id, "") if cls_id is not None else ""
            target = muddy if cls_name in {"muddy", "muddy_water", "brown_water"} else regular
            target[resized >= 0.5] = 255
        return regular, muddy

    def _segment_roboflow(self, roi_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import requests

        regular = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
        muddy = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", roi_frame)
        if not ok:
            return regular, muddy
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        endpoint = self.config.roboflow_endpoint.strip("/")
        url = f"{self.config.roboflow_api_url.rstrip('/')}/{endpoint}"
        response = requests.post(
            url,
            params={"api_key": self.config.roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")},
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        response.raise_for_status()
        for pred in response.json().get("predictions", []):
            confidence = float(pred.get("confidence", 0.0) or 0.0)
            if confidence < self.config.roboflow_confidence:
                continue
            points = pred.get("points") or []
            if len(points) < 3:
                continue
            polygon = np.array(
                [[int(point.get("x", 0)), int(point.get("y", 0))] for point in points],
                dtype=np.int32,
            )
            cls_name = str(pred.get("class", "")).lower()
            target = muddy if cls_name in {"muddy", "muddy_water", "brown_water"} else regular
            cv2.fillPoly(target, [polygon], 255)
        return regular, muddy

    def _surface_mask_for_shape(self, shape: tuple[int, ...]) -> Optional[np.ndarray]:
        if not self.config.surface_polygons:
            return None
        height, width = shape[:2]
        cache_shape = (height, width)
        if self._surface_mask_shape == cache_shape and self._surface_mask is not None:
            return self._surface_mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in self.config.surface_polygons:
            if len(polygon) < 3:
                continue
            points = np.array(
                [
                    [
                        int(np.clip(x, 0.0, 1.0) * width),
                        int(np.clip(y, 0.0, 1.0) * height),
                    ]
                    for x, y in polygon
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [points], 255)
        self._surface_mask_shape = cache_shape
        self._surface_mask = mask
        return mask


def roi_to_pixels(shape: tuple[int, ...], roi: Optional[Roi]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    if roi is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = roi
    x0_i = int(np.clip(x0, 0.0, 1.0) * width)
    y0_i = int(np.clip(y0, 0.0, 1.0) * height)
    x1_i = int(np.clip(x1, 0.0, 1.0) * width)
    y1_i = int(np.clip(y1, 0.0, 1.0) * height)
    if x1_i <= x0_i or y1_i <= y0_i:
        raise ValueError("ROI must satisfy x0 < x1 and y0 < y1")
    return x0_i, y0_i, x1_i, y1_i


def postprocess_mask(mask: np.ndarray, min_area: int, kernel_size: int, border_margin: int) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if border_margin > 0:
        mask[:border_margin, :] = 0
        mask[-border_margin:, :] = 0
        mask[:, :border_margin] = 0
        mask[:, -border_margin:] = 0
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return filter_small_components(mask, min_area)


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def contours_from_mask(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
        kept.append(cv2.approxPolyDP(contour, epsilon, True))
    return kept


def edge_mask_from_contours(
    shape: tuple[int, int],
    contours: list[np.ndarray],
    thickness: int,
) -> np.ndarray:
    edge_mask = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(
        edge_mask,
        contours,
        isClosed=True,
        color=255,
        thickness=max(1, thickness),
        lineType=cv2.LINE_AA,
    )
    return edge_mask


def surface_border_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    kernel_size = max(3, margin * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner = cv2.erode(mask, kernel)
    outer = cv2.dilate(mask, kernel)
    return cv2.subtract(outer, inner)


def blend_color(pixels: np.ndarray, color: ColorBGR, alpha: float) -> np.ndarray:
    color_arr = np.array(color, dtype=np.float32)
    return (pixels.astype(np.float32) * (1 - alpha) + color_arr * alpha).astype(np.uint8)


def normalize_class_filter(classes: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not classes:
        return None
    normalized = {str(item).lower().strip() for item in classes if str(item).strip()}
    return normalized or None


def parse_roi(value: Optional[str]) -> Optional[Roi]:
    if value is None or not value.strip():
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must have four comma-separated values: x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI values must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def parse_polygons(value: Optional[str]) -> Optional[tuple[Polygon, ...]]:
    if value is None or not value.strip():
        return None
    polygons: list[Polygon] = []
    for raw_polygon in value.split(";"):
        points: list[Point] = []
        for raw_point in raw_polygon.strip().split():
            parts = [float(part.strip()) for part in raw_point.split(",")]
            if len(parts) != 2:
                raise ValueError("Polygon points must be x,y pairs separated by spaces")
            x, y = parts
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("Polygon coordinates must be normalized values from 0 to 1")
            points.append((x, y))
        if len(points) < 3:
            raise ValueError("Each polygon must contain at least three points")
        polygons.append(tuple(points))
    return tuple(polygons) or None


def surface_preset_polygons(name: Optional[str]) -> Optional[tuple[Polygon, ...]]:
    if name is None or name.strip().lower() in {"", "none"}:
        return None
    normalized = name.strip().lower()
    if normalized != "yeongildae":
        raise ValueError("surface preset must be 'none' or 'yeongildae'")
    return (
        (
            (0.00, 0.40),
            (0.14, 0.40),
            (0.24, 0.36),
            (0.39, 0.36),
            (0.47, 0.39),
            (0.55, 0.38),
            (0.60, 0.31),
            (0.66, 0.24),
            (0.77, 0.20),
            (1.00, 0.16),
            (1.00, 1.00),
            (0.00, 1.00),
        ),
    )


def parse_bgr(value: str) -> ColorBGR:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Color must have three comma-separated BGR integers")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)  # type: ignore[return-value]
