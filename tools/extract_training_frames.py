from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract still frames for water-segmentation labeling.")
    parser.add_argument("--source", required=True, help="Video path, camera index, or RTSP URL")
    parser.add_argument("--output-dir", default="datasets/water_seg/raw_frames")
    parser.add_argument("--start-ms", type=float, default=0.0)
    parser.add_argument("--end-ms", type=float, default=0.0)
    parser.add_argument("--every-sec", type=float, default=2.0)
    parser.add_argument("--prefix", default="frame")
    parser.add_argument(
        "--seek-mode",
        action="store_true",
        help="Seek by timestamp instead of stepping through decoded frames. Useful for MP4s with broken FPS metadata.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"failed to open source: {args.source}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seek_mode:
        saved = extract_by_seek(cap, out_dir, args)
        cap.release()
        print(f"saved {saved} frames to {out_dir.resolve()}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pending_frame, pending_ms = read_seek_frame(cap, args.start_ms)
    frame_step = max(1, int(round(fps * args.every_sec)))
    saved = 0
    seen = 0
    while True:
        if pending_frame is not None:
            ok = True
            frame = pending_frame
            pending_frame = None
            pos_ms = pending_ms if pending_ms is not None else cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
            ok, frame = cap.read()
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ok:
            break
        if args.end_ms > 0 and pos_ms > args.end_ms:
            break
        if seen % frame_step == 0:
            filename = out_dir / f"{args.prefix}_{saved:05d}_{int(pos_ms):08d}ms.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
            if args.max_images > 0 and saved >= args.max_images:
                break
        seen += 1
    cap.release()
    print(f"saved {saved} frames to {out_dir.resolve()}")
    return 0


def extract_by_seek(cap: cv2.VideoCapture, out_dir: Path, args) -> int:
    seek_ms = max(0.0, args.start_ms)
    step_ms = max(1.0, args.every_sec * 1000.0)
    saved = 0
    while True:
        if args.end_ms > 0 and seek_ms > args.end_ms:
            break
        frame, pos_ms = read_seek_frame(cap, seek_ms)
        seek_ms += step_ms
        if frame is None or pos_ms is None:
            break
        filename = out_dir / f"{args.prefix}_{saved:05d}_{int(pos_ms):08d}ms.jpg"
        cv2.imwrite(str(filename), frame)
        saved += 1
        if args.max_images > 0 and saved >= args.max_images:
            break
    return saved


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
