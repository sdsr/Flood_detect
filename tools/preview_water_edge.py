from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.water_edge_segmenter import (
    EdgeConfig,
    WaterEdgeSegmenter,
    parse_bgr,
    parse_polygons,
    parse_roi,
    surface_preset_polygons,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or render flood-water edge lines on a road flooding video."
    )
    parser.add_argument("--source", required=True, help="Video file, camera index, or RTSP URL")
    parser.add_argument("--output", default=None, help="Optional MP4 path to write the overlaid demo")
    parser.add_argument(
        "--backend",
        choices=("heuristic", "yolo", "yolo11", "yolo26", "roboflow", "river"),
        default="heuristic",
    )
    parser.add_argument("--seg-model", default=None, help="YOLO/YOLO11/YOLO26 segmentation .pt path")
    parser.add_argument("--roi", default=None, help="Normalized ROI as x0,y0,x1,y1")
    parser.add_argument("--no-window", action="store_true", help="Render without opening cv2.imshow")
    parser.add_argument("--start-ms", type=float, default=0.0, help="Seek to this timestamp before reading")
    parser.add_argument("--end-ms", type=float, default=0.0, help="Stop when this timestamp is reached; 0 disables")
    parser.add_argument("--seek-step-sec", type=float, default=5.0, help="Interactive A/D seek step in seconds")
    parser.add_argument("--big-seek-step-sec", type=float, default=30.0, help="Interactive Z/X seek step in seconds")
    parser.add_argument(
        "--seekbar-range-sec",
        type=float,
        default=300.0,
        help="Fallback seek-bar range in seconds when video duration metadata is unavailable",
    )
    parser.add_argument("--pause", action="store_true", help="Start preview paused")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means full video")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=1.0,
        help="Resize the source frame before segmentation, display, and optional output.",
    )
    parser.add_argument("--display-scale", type=float, default=1.0, help="Preview window scale")
    parser.add_argument("--line-color", default=None, help="Deprecated alias for --water-color")
    parser.add_argument("--water-color", default="255,255,0", help="BGR color for regular/reflective water")
    parser.add_argument("--muddy-color", default="0,170,255", help="BGR color for muddy/brown water")
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--mask-alpha", type=float, default=0.16)
    parser.add_argument(
        "--edge-mode",
        choices=("layers", "combined"),
        default="layers",
        help="Draw separate regular/muddy edges or one combined water outline.",
    )
    parser.add_argument("--edge-color", default=None, help="BGR color for --edge-mode combined")
    parser.add_argument(
        "--edge-smooth-ratio",
        type=float,
        default=0.0025,
        help="Contour simplification ratio. Higher values draw smoother, less jagged outlines.",
    )
    parser.add_argument(
        "--edge-curve-iterations",
        type=int,
        default=0,
        help="Round contour corners while keeping more boundary points. Try 1-2 for smoother live edges.",
    )
    parser.add_argument(
        "--edge-bridge-pixels",
        type=int,
        default=0,
        help="Close small mask gaps before drawing edge lines, without changing the model output.",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="Run segmentation on a downscaled frame, then draw the mask on the original frame.",
    )
    parser.add_argument("--show-roi-border-edges", action="store_true")
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument(
        "--max-component-aspect",
        type=float,
        default=0.0,
        help="Drop long, thin mask components with a bounding-box aspect ratio above this value; 0 disables.",
    )
    parser.add_argument("--morph-kernel", type=int, default=9)
    parser.add_argument("--border-margin", type=int, default=8)
    parser.add_argument("--sat-max", type=int, default=95)
    parser.add_argument("--value-percentile", type=float, default=58.0)
    parser.add_argument("--min-value", type=int, default=45)
    parser.add_argument("--max-value", type=int, default=235)
    parser.add_argument("--texture-std-max", type=float, default=24.0)
    parser.add_argument("--disable-muddy-water", action="store_true")
    parser.add_argument("--muddy-hue-min", type=int, default=8)
    parser.add_argument("--muddy-hue-max", type=int, default=42)
    parser.add_argument("--muddy-sat-min", type=int, default=22)
    parser.add_argument("--muddy-sat-max", type=int, default=175)
    parser.add_argument("--muddy-value-min", type=int, default=55)
    parser.add_argument("--muddy-value-max", type=int, default=225)
    parser.add_argument("--muddy-texture-std-max", type=float, default=34.0)
    parser.add_argument(
        "--muddy-loose",
        action="store_true",
        help="Add wider HSV/Lab muddy-water candidates for pale or reflective brown water.",
    )
    parser.add_argument(
        "--muddy-expand-pixels",
        type=int,
        default=0,
        help="Grow muddy candidates inside the Roboflow flood mask by this many pixels.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument(
        "--hybrid-muddy",
        action="store_true",
        help="Fuse YOLO masks with the muddy-water color heuristic to recover missed brown water.",
    )
    parser.add_argument(
        "--yolo-classes",
        default=None,
        help="Comma-separated class names or ids to keep. Default keeps all masks.",
    )
    parser.add_argument(
        "--surface-preset",
        default="none",
        choices=("none", "yeongildae", "yeongildae-road"),
        help="Valid-surface mask. Use yeongildae-road when storefront/building areas are included.",
    )
    parser.add_argument(
        "--surface-polygon",
        default=None,
        help="Normalized polygon points, e.g. '0,0.4 1,0.2 1,1 0,1'. Use ';' for multiple polygons.",
    )
    parser.add_argument("--show-surface-border-edges", action="store_true")
    parser.add_argument("--roboflow-api-key", default=None)
    parser.add_argument("--roboflow-endpoint", default="river-flood-detection/5")
    parser.add_argument("--roboflow-api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--roboflow-confidence", type=float, default=0.25)
    parser.add_argument("--roboflow-overlap", type=int, default=30)
    parser.add_argument(
        "--roboflow-mask-mode",
        choices=("all", "muddy-priority", "muddy-only"),
        default="all",
        help="Post-filter Roboflow masks with the muddy-water heuristic.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    yolo_classes = None
    if args.yolo_classes:
        yolo_classes = tuple(part.strip() for part in args.yolo_classes.split(",") if part.strip())

    try:
        surface_polygons = merge_polygons(
            surface_preset_polygons(args.surface_preset),
            parse_polygons(args.surface_polygon),
        )
        config = EdgeConfig(
            backend=args.backend,
            roi=parse_roi(args.roi),
            water_color=parse_bgr(args.line_color or args.water_color),
            muddy_color=parse_bgr(args.muddy_color),
            line_thickness=args.line_thickness,
            mask_alpha=max(0.0, min(1.0, args.mask_alpha)),
            edge_mode=args.edge_mode,
            edge_color=parse_bgr(args.edge_color) if args.edge_color else None,
            edge_smooth_ratio=max(0.0, args.edge_smooth_ratio),
            edge_curve_iterations=max(0, args.edge_curve_iterations),
            edge_bridge_pixels=max(0, args.edge_bridge_pixels),
            process_scale=max(0.1, min(1.0, args.process_scale)),
            suppress_roi_border_edges=not args.show_roi_border_edges,
            min_area=args.min_area,
            max_component_aspect=max(0.0, args.max_component_aspect),
            morph_kernel=args.morph_kernel,
            border_margin=args.border_margin,
            sat_max=args.sat_max,
            value_percentile=args.value_percentile,
            min_value=args.min_value,
            max_value=args.max_value,
            texture_std_max=args.texture_std_max,
            muddy_water_enabled=not args.disable_muddy_water,
            muddy_hue_min=args.muddy_hue_min,
            muddy_hue_max=args.muddy_hue_max,
            muddy_sat_min=args.muddy_sat_min,
            muddy_sat_max=args.muddy_sat_max,
            muddy_value_min=args.muddy_value_min,
            muddy_value_max=args.muddy_value_max,
            muddy_texture_std_max=args.muddy_texture_std_max,
            muddy_loose=args.muddy_loose,
            muddy_expand_pixels=max(0, args.muddy_expand_pixels),
            yolo_conf=args.yolo_conf,
            yolo_imgsz=args.yolo_imgsz,
            yolo_device=args.yolo_device,
            yolo_classes=yolo_classes,
            hybrid_muddy=args.hybrid_muddy,
            surface_polygons=surface_polygons,
            suppress_surface_border_edges=not args.show_surface_border_edges,
            roboflow_api_key=args.roboflow_api_key,
            roboflow_endpoint=args.roboflow_endpoint,
            roboflow_api_url=args.roboflow_api_url,
            roboflow_confidence=args.roboflow_confidence,
            roboflow_overlap=args.roboflow_overlap,
            roboflow_mask_mode=args.roboflow_mask_mode,
        )
        segmenter = WaterEdgeSegmenter(config, model_path=args.seg_model)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: failed to open source: {args.source}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_scale = max(0.1, min(1.0, args.frame_scale))
    output_width = max(1, int(round(width * frame_scale)))
    output_height = max(1, int(round(height * frame_scale)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"source opened: {width}x{height}, fps={fps:.2f}, frames={total}")
    if frame_scale < 0.999:
        print(f"frame scale: {frame_scale:.2f} -> {output_width}x{output_height}")
    if args.start_ms > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_ms)
        print(f"seek: start_ms={args.start_ms:.0f}")

    writer = None
    output_path = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps / max(1, args.frame_stride),
            (output_width, output_height),
        )
        if not writer.isOpened():
            print(f"ERROR: failed to open output writer: {output_path}", file=sys.stderr)
            return 3

    seekbar_scale = 10
    seekbar_name = "seek sec x10"
    seekbar_target_ms = None
    seekbar_updating = False
    seekbar_max = max(1, int(video_duration_seconds(total, fps, args.seekbar_range_sec) * seekbar_scale))

    if not args.no_window:
        cv2.namedWindow("Water edge demo", cv2.WINDOW_NORMAL)

        def on_seekbar(value: int) -> None:
            nonlocal seekbar_target_ms, seekbar_updating
            if seekbar_updating:
                return
            seekbar_target_ms = (value / seekbar_scale) * 1000.0

        cv2.createTrackbar(seekbar_name, "Water edge demo", 0, seekbar_max, on_seekbar)
        print("controls: slider=seek, Space=pause/resume, A/D=-/+seek, Z/X=-/+big seek, Q/Esc=quit")

    processed = 0
    read_count = 0
    current_ms = max(0.0, args.start_ms)
    paused = bool(args.pause)
    started = time.time()
    try:
        pending_frame, pending_ms = read_seek_frame(cap, args.start_ms)
        while True:
            if seekbar_target_ms is not None:
                pending_frame, pending_ms = read_seek_frame(cap, seekbar_target_ms)
                current_ms = pending_ms if pending_ms is not None else seekbar_target_ms
                seekbar_target_ms = None
            from_pending = False
            if pending_frame is not None:
                ok = True
                frame = pending_frame
                if pending_ms is not None:
                    current_ms = pending_ms
                pending_frame = None
                pending_ms = None
                from_pending = True
            else:
                if paused and not args.no_window:
                    key = cv2.waitKey(30) & 0xFF
                    action = handle_key(key, current_ms, args)
                    if action == "quit":
                        break
                    if isinstance(action, float):
                        pending_frame, pending_ms = read_seek_frame(cap, action)
                    elif action == "toggle_pause":
                        paused = not paused
                    continue
                ok, frame = cap.read()
                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not ok:
                if read_count == 0:
                    print(
                        "ERROR: failed to read frames. "
                        "Try a larger --start-ms value if the video has a damaged opening GOP.",
                        file=sys.stderr,
                    )
                break
            if args.end_ms > 0 and current_ms >= args.end_ms:
                break
            read_count += 1
            if not from_pending and args.frame_stride > 1 and (read_count - 1) % args.frame_stride != 0:
                continue
            if frame_scale < 0.999:
                frame = resize_for_display(frame, frame_scale)

            try:
                output, _mask, contours = segmenter.draw(frame)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 5
            processed += 1
            draw_hud(output, processed, len(contours), args.backend, current_ms, paused)
            if not args.no_window:
                seekbar_value = min(seekbar_max, max(0, int((current_ms / 1000.0) * seekbar_scale)))
                if cv2.getTrackbarPos(seekbar_name, "Water edge demo") != seekbar_value:
                    seekbar_updating = True
                    cv2.setTrackbarPos(seekbar_name, "Water edge demo", seekbar_value)
                    seekbar_updating = False

            if writer is not None:
                writer.write(output)

            if not args.no_window:
                shown = resize_for_display(output, args.display_scale)
                cv2.imshow("Water edge demo", shown)
                key = cv2.waitKey(1) & 0xFF
                action = handle_key(key, current_ms, args)
                if action == "quit":
                    break
                if action == "toggle_pause":
                    paused = not paused
                elif isinstance(action, float):
                    pending_frame, pending_ms = read_seek_frame(cap, action)

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - started)
    print(f"processed={processed}, elapsed={elapsed:.2f}s, throughput={processed / elapsed:.2f} fps")
    if output_path is not None:
        print(f"output saved: {output_path.resolve()}")
    return 0


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def merge_polygons(*items):
    merged = []
    for item in items:
        if item:
            merged.extend(item)
    return tuple(merged) or None


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


def video_duration_seconds(total_frames: int, fps: float, fallback_sec: float) -> float:
    if total_frames > 0 and 1 <= fps <= 240:
        return max(1.0, total_frames / fps)
    return max(1.0, fallback_sec)


def handle_key(key: int, current_ms: float, args) -> str | float | None:
    if key == 255:
        return None
    if key in (ord("q"), ord("Q"), 27):
        return "quit"
    if key in (ord(" "), ord("p"), ord("P")):
        return "toggle_pause"
    if key in (ord("a"), ord("A")):
        return max(0.0, current_ms - args.seek_step_sec * 1000.0)
    if key in (ord("d"), ord("D")):
        return current_ms + args.seek_step_sec * 1000.0
    if key in (ord("z"), ord("Z")):
        return max(0.0, current_ms - args.big_seek_step_sec * 1000.0)
    if key in (ord("x"), ord("X")):
        return current_ms + args.big_seek_step_sec * 1000.0
    return None


def resize_for_display(frame, scale: float):
    if abs(scale - 1.0) < 0.001:
        return frame
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_hud(frame, processed: int, contour_count: int, backend: str, current_ms: float, paused: bool) -> None:
    status = "paused" if paused else "playing"
    text = (
        f"Water edge | {status} | t={format_ms(current_ms)} | "
        f"frame={processed} | contours={contour_count}"
    )
    controls = "Slider seek | Space pause | A/D seek 5s | Z/X seek 30s | Q quit"
    cv2.rectangle(frame, (8, 8), (930, 68), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        controls,
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def format_ms(value: float) -> str:
    total = max(0, int(value // 1000))
    minutes = total // 60
    seconds = total % 60
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
