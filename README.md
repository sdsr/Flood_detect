# Water Edge Demo

도로 침수 영상에서 물 영역의 가장자리를 선으로 표시하는 실험용 프로젝트입니다. 작업 위치는 이 폴더입니다.

```text
D:\My_backup\workspace\flood_test
```

`D:\SIS\202605\Flood_detector-main`은 샘플 영상을 읽는 용도로만 사용합니다.

## 현재 구현

- `heuristic`: 별도 모델 없이 OpenCV 색상/질감 규칙으로 물 경계 표시
- `yolo11`: 로컬 YOLO11 segmentation `.pt`로 물 경계 표시
- `river`: Roboflow Universe의 river/flood 모델을 hosted API로 호출
- 실시간 창에서 `Space`, `A/D`, `Z/X` 키 이동 지원
- 실시간 창 상단 seek bar로 영상 구간 이동 지원
- 일반/반사 물은 청록색, 갈색/탁수 물은 주황색으로 분리 표시
- 기본 실행/학습은 전체 프레임 감지 기준이며, ROI는 필요할 때만 선택적으로 지정
- `--surface-preset yeongildae`: 샘플 CCTV에서 하늘/건물/유리 영역을 제외하고 도로/보도 표면만 후보로 사용
- `--hybrid-muddy`: YOLO가 놓친 흙탕물 후보를 색상 기반 규칙으로 보강

## 가상환경

이 프로젝트의 의존성은 전역 Python이 아니라 `.venv`에 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-yolo.txt
```

기본 OpenCV 휴리스틱만 쓸 때는 `requirements.txt`만 설치해도 됩니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## 바로 실행: YOLO11 데모 pt

현재 생성된 전체 프레임 데모 가중치가 있습니다.

```text
models\flood_seg\best.pt
```

주의: 이 파일은 사람이 직접 라벨링한 정식 모델이 아니라, 현재 OpenCV 휴리스틱 결과를 pseudo-label로 만들어 학습한 데모 모델입니다. 시연과 구조 검증용으로는 쓸 수 있지만, 운영 정확도는 실제 침수 프레임을 라벨링해서 다시 학습해야 합니다.
이전 ROI 기준 모델은 `models\flood_seg\best_roi_backup.pt`에 백업되어 있습니다.

실시간 미리보기:

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py `
  --source "D:\SIS\202605\Flood_detector-main\3_학파_영일대.mp4" `
  --backend yolo11 `
  --seg-model .\models\flood_seg\best.pt `
  --start-ms 1000 `
  --yolo-imgsz 512 `
  --yolo-conf 0.35 `
  --yolo-device cpu `
  --min-area 2500 `
  --surface-preset yeongildae `
  --hybrid-muddy `
  --muddy-hue-min 5 `
  --muddy-hue-max 55 `
  --muddy-sat-min 10 `
  --muddy-sat-max 215 `
  --muddy-value-min 35 `
  --muddy-value-max 235 `
  --muddy-texture-std-max 52
```

조작:

- 상단 seek bar: 원하는 구간으로 이동
- `Space` 또는 `P`: 일시정지/재생
- `A` / `D`: 5초 뒤/앞
- `Z` / `X`: 30초 뒤/앞
- `Q` 또는 `Esc`: 종료

영상 길이 메타데이터가 깨진 파일은 seek bar 범위를 자동으로 못 잡을 수 있습니다. 그때는 범위를 직접 늘립니다.

```powershell
--seekbar-range-sec 900
```

## MP4로 렌더링

창 없이 결과 영상을 저장하려면 `--output`과 `--no-window`를 씁니다.

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py `
  --source "D:\SIS\202605\Flood_detector-main\3_학파_영일대.mp4" `
  --backend yolo11 `
  --seg-model .\models\flood_seg\best.pt `
  --output .\outputs\water_edge_yolo11_surface_hybrid_demo.mp4 `
  --no-window `
  --start-ms 1000 `
  --max-frames 120 `
  --frame-stride 4 `
  --yolo-imgsz 512 `
  --yolo-conf 0.35 `
  --yolo-device cpu `
  --min-area 2500 `
  --surface-preset yeongildae `
  --hybrid-muddy `
  --muddy-hue-min 5 `
  --muddy-hue-max 55 `
  --muddy-sat-min 10 `
  --muddy-sat-max 215 `
  --muddy-value-min 35 `
  --muddy-value-max 235 `
  --muddy-texture-std-max 52
```

검증용 결과:

```text
outputs\water_edge_yolo11_full_demo.mp4
outputs\water_edge_yolo11_full_demo_frame40.jpg
outputs\water_edge_yolo11_full_conf035_demo.mp4
outputs\water_edge_yolo11_full_conf035_demo_frame40.jpg
outputs\water_edge_yolo11_surface_hybrid_demo.mp4
outputs\water_edge_yolo11_surface_hybrid_demo_frame40.jpg
```

## OpenCV 휴리스틱 실행

모델 없이 빠르게 확인할 때 사용합니다.

```powershell
.\.venv\Scripts\python.exe .\tools\preview_water_edge.py `
  --source "D:\SIS\202605\Flood_detector-main\3_학파_영일대.mp4" `
  --start-ms 1000 `
  --min-area 6500 `
  --morph-kernel 15 `
  --texture-std-max 22 `
  --sat-max 92 `
  --value-percentile 57 `
  --max-value 222 `
  --muddy-hue-min 8 `
  --muddy-hue-max 45 `
  --muddy-sat-min 18 `
  --muddy-sat-max 185 `
  --muddy-value-min 50 `
  --muddy-value-max 225 `
  --muddy-texture-std-max 36
```

기본은 전체 프레임 감지입니다. 특정 도로 구역만 보고 싶을 때만 `--roi 0.30,0.30,1.0,0.96`처럼 ROI를 추가합니다.
하늘이나 건물까지 감지되는 경우에는 ROI 대신 `--surface-preset yeongildae` 또는 직접 만든 `--surface-polygon`을 쓰는 편이 낫습니다. 이건 flood detector의 작은 ROI가 아니라, 고정 CCTV에서 물이 존재할 수 있는 도로/보도 표면만 남기는 마스크입니다.

## river 모델 사용

`--backend river`는 Roboflow hosted API를 사용합니다. 로컬 `.pt` 파일이 아니라 API 키가 필요합니다.

```powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"

.\.venv\Scripts\python.exe .\tools\preview_water_edge.py `
  --source "D:\SIS\202605\Flood_detector-main\3_학파_영일대.mp4" `
  --backend river `
  --roboflow-api-url "https://serverless.roboflow.com" `
  --roboflow-endpoint "river-flood-detection/5" `
  --roboflow-confidence 0.25 `
  --start-ms 1000
```

Roboflow Universe 페이지나 API 화면에서 실제 endpoint/version이 다르면 `--roboflow-endpoint`를 그 값으로 바꾸면 됩니다. API 키가 없으면 다음처럼 명확히 종료됩니다.
구버전 instance segmentation endpoint를 써야 하는 경우에는 `--roboflow-api-url "https://outline.roboflow.com"`로 바꿔 실행할 수 있습니다.

```text
ERROR: Roboflow backend requires --roboflow-api-key or ROBOFLOW_API_KEY
```

## 데모 pt 다시 만들기

현재 휴리스틱 결과를 pseudo-label로 YOLO segmentation 데이터셋을 만든 뒤 YOLO11을 fine-tuning합니다.

```powershell
.\.venv\Scripts\python.exe .\tools\build_pseudo_yolo_dataset.py `
  --source "D:\SIS\202605\Flood_detector-main\3_학파_영일대.mp4" `
  --output-dir .\datasets\water_seg_full `
  --clean `
  --start-ms 1000 `
  --every-sec 1 `
  --max-images 80 `
  --val-ratio 0.2
```

```powershell
.\.venv\Scripts\python.exe .\tools\train_yolo11_seg.py `
  --data .\datasets\water_seg_full\data.yaml `
  --model yolo11n-seg.pt `
  --epochs 25 `
  --imgsz 512 `
  --batch 2 `
  --device cpu `
  --project .\runs\water_seg `
  --name yolo11n_pseudo_full_frame
```

학습 후 `best.pt`를 복사합니다.

```powershell
Copy-Item .\runs\segment\runs\water_seg\yolo11n_pseudo_full_frame\weights\best.pt .\models\flood_seg\best.pt -Force
```

## 공개 가중치에 대한 메모

CCTV 도로 침수의 "물 가장자리"에 바로 맞는 범용 공개 `.pt`는 흔치 않습니다. 보통은 다음 중 하나로 갑니다.

- 지금처럼 pseudo-label 데모 모델로 빠르게 구조 검증
- Roboflow Universe 같은 공개 hosted 모델/API로 비교
- 실제 영상 프레임을 직접 polygon segmentation 라벨링 후 YOLO11/SegFormer fine-tuning

운영 목적이면 세 번째가 가장 안정적입니다.
