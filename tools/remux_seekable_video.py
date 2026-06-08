from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a seek-friendly MP4 for OpenCV preview. This is useful for "
            "CCTV files whose container metadata makes cv2.CAP_PROP_POS_MSEC seek "
            "land at the wrong timestamp."
        )
    )
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument(
        "--transcode",
        action="store_true",
        help="Re-encode with H.264 instead of fast stream copy. Slower, but can fix more damaged files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        return 2
    if output.exists() and not args.overwrite:
        print(f"ERROR: output already exists: {output}", file=sys.stderr)
        print("Pass --overwrite to replace it.", file=sys.stderr)
        return 3
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio_ffmpeg
    except ImportError:
        print(
            "ERROR: imageio-ffmpeg is not installed. Run: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-yolo.txt",
            file=sys.stderr,
        )
        return 4

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if args.transcode:
        codec_args = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        codec_args = ["-c:v", "copy"]

    command = [
        ffmpeg,
        "-hide_banner",
        "-y" if args.overwrite else "-n",
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+genpts",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        *codec_args,
        "-movflags",
        "+faststart",
        str(output),
    ]
    print("running ffmpeg remux...")
    print(f"source: {source}")
    print(f"output: {output}")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
