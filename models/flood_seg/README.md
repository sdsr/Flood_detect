# Flood Segmentation Models

These checkpoints are demo segmentation models trained from heuristic
pseudo-labels, not production-quality human-labeled models.

YOLO11 demo checkpoint:

```text
models/flood_seg/best.pt
```

YOLO26 demo checkpoint:

```text
models/flood_seg/yolo26n_best.pt
```

YOLO26 full-video pseudo-label checkpoint:

```text
models/flood_seg/yolo26n_full20_best.pt
```

Expected classes:

- `water`
- `muddy_water`

Run YOLO11 with:

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py --backend yolo11 --seg-model .\models\flood_seg\best.pt --source "video.mp4" --yolo-conf 0.35 --min-area 2500 --surface-preset yeongildae --hybrid-muddy
```

Run YOLO26 with:

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py --backend yolo26 --seg-model .\models\flood_seg\yolo26n_best.pt --source "video.mp4" --yolo-conf 0.05 --yolo-classes muddy_water --max-component-aspect 5 --edge-mode combined --edge-smooth-ratio 0.006 --edge-bridge-pixels 12 --line-thickness 4 --mask-alpha 0
```

Run the full-video YOLO26 model around the late muddy-water section:

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py `
  --source "video.mp4" `
  --backend yolo26 `
  --seg-model .\models\flood_seg\yolo26n_full20_best.pt `
  --start-ms 1050000 `
  --frame-scale 0.4 `
  --yolo-imgsz 512 `
  --yolo-conf 0.15 `
  --yolo-device cpu `
  --min-area 500 `
  --morph-kernel 7 `
  --surface-preset yeongildae-road `
  --yolo-classes muddy_water `
  --edge-mode combined `
  --edge-smooth-ratio 0.006 `
  --edge-bridge-pixels 12 `
  --line-thickness 4 `
  --mask-alpha 0 `
  --seekbar-range-sec 1200
```

Training summary:

```text
dataset: datasets/water_seg_full20
base: yolo26n-seg.pt
epochs: 30
image size: 512
device: CPU
final mask mAP50: 0.488
muddy_water mask mAP50: 0.683
```
