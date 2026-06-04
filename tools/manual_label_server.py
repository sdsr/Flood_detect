from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "water",
    1: "muddy_water",
}
CLASS_COLORS = {
    0: "#16d9e8",
    1: "#ffae00",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser-based YOLO segmentation labeler.")
    parser.add_argument("--dataset", default="datasets/yeongildae_manual_5s")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


class LabelStore:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir / "images"
        self.label_dir = dataset_dir / "labels"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"image dir not found: {self.image_dir}")
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise FileNotFoundError(f"no images found in: {self.image_dir}")
        self._shape_cache: dict[str, tuple[int, int]] = {}
        self.ensure_empty_label_files()
        self.write_data_yaml()

    def ensure_empty_label_files(self) -> None:
        for image_path in self.images:
            label_path = self.label_path_for(image_path)
            if not label_path.exists():
                label_path.write_text("", encoding="utf-8")

    def write_data_yaml(self) -> None:
        yaml_path = self.dataset_dir / "data.yaml"
        path_text = str(self.dataset_dir.resolve()).replace("\\", "/")
        yaml_path.write_text(
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

    def label_path_for(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def get_image(self, index: int) -> Path:
        return self.images[self.clamp_index(index)]

    def clamp_index(self, index: int) -> int:
        return max(0, min(len(self.images) - 1, index))

    def image_shape(self, image_path: Path) -> tuple[int, int]:
        key = image_path.name
        if key in self._shape_cache:
            return self._shape_cache[key]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = frame.shape[:2]
        self._shape_cache[key] = (width, height)
        return width, height

    def read_labels(self, index: int) -> dict:
        image_path = self.get_image(index)
        label_path = self.label_path_for(image_path)
        width, height = self.image_shape(image_path)
        labels = []
        if label_path.exists():
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    coords = [float(part) for part in parts[1:]]
                except ValueError:
                    continue
                points = []
                for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
                    points.append(
                        [
                            max(0.0, min(1.0, x_norm)),
                            max(0.0, min(1.0, y_norm)),
                        ]
                    )
                if len(points) >= 3:
                    labels.append({"class_id": class_id, "points": points})
        return {
            "index": self.clamp_index(index),
            "count": len(self.images),
            "name": image_path.name,
            "width": width,
            "height": height,
            "classes": [
                {"id": idx, "name": name, "color": CLASS_COLORS[idx]}
                for idx, name in CLASS_NAMES.items()
            ],
            "labels": labels,
            "labeled_count": self.labeled_count(),
        }

    def write_labels(self, index: int, labels: list[dict]) -> dict:
        image_path = self.get_image(index)
        label_path = self.label_path_for(image_path)
        lines = []
        for label in labels:
            try:
                class_id = int(label.get("class_id", 1))
            except (TypeError, ValueError):
                class_id = 1
            if class_id not in CLASS_NAMES:
                continue
            points = label.get("points") or []
            cleaned = []
            for point in points:
                if not isinstance(point, list | tuple) or len(point) != 2:
                    continue
                try:
                    x = max(0.0, min(1.0, float(point[0])))
                    y = max(0.0, min(1.0, float(point[1])))
                except (TypeError, ValueError):
                    continue
                cleaned.append((x, y))
            if len(cleaned) < 3:
                continue
            flattened = " ".join(f"{value:.6f}" for point in cleaned for value in point)
            lines.append(f"{class_id} {flattened}")
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        return self.read_labels(index)

    def labeled_count(self) -> int:
        count = 0
        for image_path in self.images:
            label_path = self.label_path_for(image_path)
            if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
                count += 1
        return count

    def stats(self) -> dict:
        class_counts = {str(idx): 0 for idx in CLASS_NAMES}
        labeled = 0
        empty = 0
        for image_path in self.images:
            label_path = self.label_path_for(image_path)
            text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
            lines = [line for line in text.splitlines() if line.strip()]
            if lines:
                labeled += 1
            else:
                empty += 1
            for line in lines:
                class_id = line.split(maxsplit=1)[0]
                if class_id in class_counts:
                    class_counts[class_id] += 1
        return {
            "images": len(self.images),
            "labeled_images": labeled,
            "empty_images": empty,
            "polygons_by_class": class_counts,
            "dataset": str(self.dataset_dir.resolve()),
        }


def make_handler(store: LabelStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.send_text(HTML_PAGE, "text/html; charset=utf-8")
                elif parsed.path == "/api/images":
                    self.send_json({"images": [path.name for path in store.images], "count": len(store.images)})
                elif parsed.path == "/api/labels":
                    self.send_json(store.read_labels(parse_index(query)))
                elif parsed.path == "/api/stats":
                    self.send_json(store.stats())
                elif parsed.path == "/image":
                    image_path = store.get_image(parse_index(query))
                    self.send_file(image_path)
                else:
                    self.send_error(404, "not found")
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, str(exc))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path != "/api/labels":
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body or "{}")
                self.send_json(store.write_labels(parse_index(query), payload.get("labels") or []))
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, str(exc))

        def send_json(self, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def send_text(self, text: str, content_type: str) -> None:
            raw = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def send_file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            raw = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def parse_index(query: dict[str, list[str]]) -> int:
    try:
        return int(query.get("index", ["0"])[0])
    except ValueError:
        return 0


HTML_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Water Labeler</title>
<style>
:root {
  color-scheme: dark;
  --bg: #15171b;
  --panel: #20242a;
  --panel2: #282d34;
  --text: #edf0f2;
  --muted: #aab2bd;
  --line: #3a414c;
  --cyan: #16d9e8;
  --orange: #ffae00;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 Arial, sans-serif; }
button, select, input { font: inherit; }
.app { display: grid; grid-template-rows: 54px 1fr; min-height: 100%; }
.topbar {
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 8px 12px;
  background: #101216;
  border-bottom: 1px solid var(--line);
}
.group { display: flex; align-items: center; gap: 6px; min-width: 0; }
.status { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.main { display: grid; grid-template-columns: 1fr 320px; min-height: 0; }
.stage {
  position: relative;
  display: grid;
  place-items: center;
  min-width: 0;
  min-height: 0;
  background: #0b0c0f;
  overflow: hidden;
}
canvas {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  display: block;
  cursor: crosshair;
}
.side {
  display: grid;
  grid-template-rows: auto auto 1fr auto;
  gap: 12px;
  min-height: 0;
  padding: 12px;
  background: var(--panel);
  border-left: 1px solid var(--line);
}
.panel {
  padding: 10px;
  background: var(--panel2);
  border: 1px solid var(--line);
  border-radius: 6px;
}
.row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
.row > * { min-width: 0; }
button {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #343a44;
  color: var(--text);
  padding: 0 10px;
  cursor: pointer;
}
button:hover { background: #3d4550; }
button.primary { border-color: #347f8a; background: #12606a; }
button.warn { border-color: #8b6622; background: #614919; }
select, input[type="number"] {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #15191f;
  color: var(--text);
  padding: 0 8px;
}
input[type="range"] { width: 100%; }
.class-toggle { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.class-toggle button { height: 38px; }
.class-toggle .active[data-class="0"] { border-color: var(--cyan); box-shadow: inset 0 0 0 2px var(--cyan); }
.class-toggle .active[data-class="1"] { border-color: var(--orange); box-shadow: inset 0 0 0 2px var(--orange); }
.list { overflow: auto; min-height: 0; }
.poly {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px;
  margin-bottom: 6px;
  background: #171b21;
  border: 1px solid var(--line);
  border-radius: 6px;
}
.poly small { color: var(--muted); }
.footer { color: var(--muted); font-size: 12px; }
@media (max-width: 900px) {
  .main { grid-template-columns: 1fr; grid-template-rows: minmax(340px, 1fr) auto; }
  .side { border-left: 0; border-top: 1px solid var(--line); grid-template-rows: auto auto auto auto; }
  .topbar { grid-template-columns: auto 1fr auto; }
  .topbar .status { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="group">
      <button id="prevBtn" title="Previous frame">Prev</button>
      <button id="nextBtn" title="Next frame">Next</button>
    </div>
    <div class="group">
      <input id="frameRange" type="range" min="0" max="0" value="0" />
      <input id="frameNumber" type="number" min="0" value="0" />
    </div>
    <div id="status" class="status">Loading</div>
    <div class="group">
      <button id="saveBtn" class="primary" title="Save labels">Save</button>
    </div>
  </div>
  <div class="main">
    <div class="stage"><canvas id="canvas"></canvas></div>
    <aside class="side">
      <div class="panel">
        <div class="class-toggle">
          <button id="class0" data-class="0">water</button>
          <button id="class1" data-class="1" class="active">muddy_water</button>
        </div>
      </div>
      <div class="panel">
        <div class="row">
          <button id="finishBtn" class="primary">Finish polygon</button>
          <button id="undoPointBtn">Undo point</button>
        </div>
        <div class="row">
          <button id="cancelBtn">Cancel</button>
          <button id="clearBtn" class="warn">Clear frame</button>
        </div>
        <div class="row">
          <button id="copyPrevBtn">Copy previous</button>
          <button id="emptySaveBtn">Save empty</button>
        </div>
      </div>
      <div id="polyList" class="list"></div>
      <div id="stats" class="footer"></div>
    </aside>
  </div>
</div>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const rangeEl = document.getElementById("frameRange");
const frameNumberEl = document.getElementById("frameNumber");
const polyListEl = document.getElementById("polyList");
const colors = {0: "#16d9e8", 1: "#ffae00"};
const names = {0: "water", 1: "muddy_water"};
let index = Number(new URLSearchParams(location.search).get("index") || 0);
let count = 0;
let labels = [];
let current = [];
let currentClass = 1;
let image = new Image();
let imageName = "";
let dirty = false;

function setClass(id) {
  currentClass = id;
  document.querySelectorAll(".class-toggle button").forEach(btn => {
    btn.classList.toggle("active", Number(btn.dataset.class) === id);
  });
}

async function loadFrame(nextIndex) {
  if (dirty && !confirm("Unsaved labels will be lost. Continue?")) return;
  index = Math.max(0, Math.min(count ? count - 1 : nextIndex, nextIndex));
  const res = await fetch(`/api/labels?index=${index}`);
  const data = await res.json();
  index = data.index;
  count = data.count;
  labels = data.labels || [];
  current = [];
  imageName = data.name;
  dirty = false;
  rangeEl.max = Math.max(0, count - 1);
  rangeEl.value = index;
  frameNumberEl.max = Math.max(0, count - 1);
  frameNumberEl.value = index;
  image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    draw();
  };
  image.src = `/image?index=${index}&t=${Date.now()}`;
  updateList();
  await loadStats();
}

async function loadStats() {
  const res = await fetch("/api/stats");
  const data = await res.json();
  statsEl.textContent = `labeled ${data.labeled_images}/${data.images} | water ${data.polygons_by_class["0"]} | muddy ${data.polygons_by_class["1"]}`;
  statusEl.textContent = `${index + 1}/${count} | ${imageName}`;
}

function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  return [
    (evt.clientX - rect.left) / rect.width,
    (evt.clientY - rect.top) / rect.height,
  ];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (image.complete) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  labels.forEach((label, idx) => drawPolygon(label.points, label.class_id, idx === -1, true));
  if (current.length) drawPolygon(current, currentClass, true, false);
}

function drawPolygon(points, classId, active, closed) {
  if (!points.length) return;
  const color = colors[classId] || "#ffffff";
  ctx.save();
  ctx.beginPath();
  points.forEach((point, idx) => {
    const x = point[0] * canvas.width;
    const y = point[1] * canvas.height;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (closed && points.length >= 3) ctx.closePath();
  if (closed && points.length >= 3) {
    ctx.globalAlpha = 0.28;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = active ? 5 : 3;
  ctx.stroke();
  points.forEach(point => {
    ctx.beginPath();
    ctx.arc(point[0] * canvas.width, point[1] * canvas.height, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#0b0c0f";
    ctx.stroke();
  });
  ctx.restore();
}

function finishPolygon() {
  if (current.length < 3) return;
  labels.push({class_id: currentClass, points: current.slice()});
  current = [];
  dirty = true;
  updateList();
  draw();
}

function updateList() {
  polyListEl.innerHTML = "";
  labels.forEach((label, idx) => {
    const item = document.createElement("div");
    item.className = "poly";
    const text = document.createElement("div");
    text.innerHTML = `<strong style="color:${colors[label.class_id] || "#fff"}">${names[label.class_id] || label.class_id}</strong><br><small>${label.points.length} points</small>`;
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.onclick = () => {
      labels.splice(idx, 1);
      dirty = true;
      updateList();
      draw();
    };
    item.appendChild(text);
    item.appendChild(del);
    polyListEl.appendChild(item);
  });
}

async function saveLabels(empty=false) {
  if (!empty) finishPolygon();
  const body = JSON.stringify({labels: empty ? [] : labels});
  const res = await fetch(`/api/labels?index=${index}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body,
  });
  const data = await res.json();
  labels = data.labels || [];
  current = [];
  dirty = false;
  updateList();
  draw();
  await loadStats();
}

async function copyPrevious() {
  if (index <= 0) return;
  const res = await fetch(`/api/labels?index=${index - 1}`);
  const data = await res.json();
  labels = JSON.parse(JSON.stringify(data.labels || []));
  current = [];
  dirty = true;
  updateList();
  draw();
}

canvas.addEventListener("click", evt => {
  current.push(canvasPoint(evt));
  dirty = true;
  draw();
});
canvas.addEventListener("dblclick", evt => {
  evt.preventDefault();
  finishPolygon();
});
document.getElementById("class0").onclick = () => setClass(0);
document.getElementById("class1").onclick = () => setClass(1);
document.getElementById("finishBtn").onclick = finishPolygon;
document.getElementById("undoPointBtn").onclick = () => { current.pop(); dirty = true; draw(); };
document.getElementById("cancelBtn").onclick = () => { current = []; draw(); };
document.getElementById("clearBtn").onclick = () => { labels = []; current = []; dirty = true; updateList(); draw(); };
document.getElementById("copyPrevBtn").onclick = copyPrevious;
document.getElementById("emptySaveBtn").onclick = () => saveLabels(true);
document.getElementById("saveBtn").onclick = () => saveLabels(false);
document.getElementById("prevBtn").onclick = () => loadFrame(index - 1);
document.getElementById("nextBtn").onclick = () => loadFrame(index + 1);
rangeEl.addEventListener("change", () => loadFrame(Number(rangeEl.value)));
frameNumberEl.addEventListener("change", () => loadFrame(Number(frameNumberEl.value)));
window.addEventListener("keydown", evt => {
  if (evt.target.tagName === "INPUT") return;
  if (evt.key === "1") setClass(0);
  if (evt.key === "2") setClass(1);
  if (evt.key === "Enter") finishPolygon();
  if (evt.key.toLowerCase() === "s") saveLabels(false);
  if (evt.key.toLowerCase() === "a") loadFrame(index - 1);
  if (evt.key.toLowerCase() === "d") loadFrame(index + 1);
  if (evt.key.toLowerCase() === "z") { current.pop(); dirty = true; draw(); }
  if (evt.key === "Escape") { current = []; draw(); }
});
setClass(1);
loadFrame(index);
</script>
</body>
</html>
"""


def main() -> int:
    args = build_parser().parse_args()
    store = LabelStore(Path(args.dataset))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}"
    print(f"labeler running: {url}")
    print(f"dataset: {store.dataset_dir.resolve()}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
