# Flood Segmentation Model

The current checkpoint is a full-frame demo YOLO11 segmentation model
trained from heuristic pseudo-labels, not a production-quality
human-labeled model.

Expected path:

```text
models/flood_seg/best.pt
```

Expected classes:

- `water`
- `muddy_water`

Run with:

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py --backend yolo11 --seg-model .\models\flood_seg\best.pt --source "video.mp4" --yolo-conf 0.35 --min-area 2500 --surface-preset yeongildae --hybrid-muddy
```
