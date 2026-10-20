# Frame cell detection - improvement plan

Status: proposal (2026-10-21). Goal: better recall/precision on real Gratheon photos
(lighting, zoom, narrow queen-rearing frames, capped honey) and much lower latency/RAM.

## 1. Why the current pipeline fails

Current pipeline = DeepBee (2020): U-Net comb mask -> Hough circles -> MobileNet per cell crop.
Root causes found in `src/DeepBee/software/detection_and_classification.py`:

| Symptom | Cause in code |
|---|---|
| Narrow frames (queen rearing), crops, phone photos | `segmentation()` force-resizes every image to 6000x4000, so any non 3:2 image is stretched. Only the single largest mask contour is kept. |
| Different zoom | `find_circles()` searches radii 5-50 px only and then keeps one "best" radius (+-10%) for the whole image. Close-ups (r > 50) and mixed scales are lost. |
| Capped honey (and partly capped brood) missed | Detection is Hough circles on the red channel with fixed thresholds. Wax caps are flat and have weak circular edges, so no circles are found there - the classifier never even sees those cells. |
| Lighting | Fixed Hough/Canny thresholds (`param1=145`, `param2=55`) + model trained on one lab-like dataset (Portugal, DSLR, controlled light). |
| Missing cells on full frames | Prod sets `MAX_CLASSIFICATION_CELLS=3200`, a Langstroth/Dadant side has 3500-5000 cells; extra cells are silently dropped. |
| Slow / RAM hungry | Upscale to 24 MP, 117 U-Net tiles, 10 Hough passes, then up to 3200 separate 224x224 MobileNet inferences on CPU in TF 2.11 / Python 3.8. |

The error is mostly in **detection**, not classification. Retraining only the classifier would not fix capped honey or zoom.

## 2. What exists (research + open data)

| Work | What it gives us | Usable? |
|---|---|---|
| DeepBee, Alves et al. 2020 (current) | 7 classes, dense crops dataset, MIT | Data yes (pretraining), approach outdated |
| FAIRHiveFrames-1K, Kulyukin et al. 2026, Sensors 26(8):2518 | 1265 frames 1920x1080, 124k boxes, 6 cell classes + frame, YOLO format, CC-BY 4.0, weights on Zenodo (doi:10.5281/zenodo.19241078) | Yes as extra training data. Labels are **sparse** (not every cell boxed), so their YOLO baselines are weak (mAP50 ~0.39, capped honey F1 ~0.26-0.29). Not a drop-in model. |
| Khokthong et al. 2025, Insects 16:575 (YOLOv11 seg) | Capped/uncapped honey as polygon areas, mAP50 0.80-0.83 at 960 px | Confirms: treat capped honey as **regions**, use high input resolution |
| Rodriguez-Lozano et al. 2024, CEA 109573 + HoneySeg 2025 | Capped honey segmentation dataset (github BeehoneySegmentation) | Yes for capped honey region data (check license) |
| buzzlogic (YOLO11, MIT) | Cells/brood pattern seg model + dataset | Worth evaluating as a quick baseline |
| Vasconcelos et al. 2025 (UTFPR, DeepBee group) | Semantic segmentation of comb cells | Supports segmentation-first approach |

Conclusion: there is **no ready model that is clearly better on our photos**. Best path is to train our own
modern detector, using public data for pretraining and our own uploaded frames for fine-tuning and evaluation.

## 3. Target design

Scale-normalized, tiled, single-pass detector:

1. **Cell pitch estimation** (cheap, classic CV): estimate cell diameter from the hexagonal lattice
   (FFT / autocorrelation peak of the comb texture). Rescale image so a cell is ~28 px. This removes
   the zoom problem and the 6000x4000 stretching for any aspect ratio.
2. **Tiled detector** (YOLO11n/s detect, or YOLO26n): 1024 px tiles with overlap, class-aware NMS
   merge. One network outputs center/box + class for every cell, including capped cells
   (hex outlines on caps are visible at the normalized scale; the model learns texture, not circles).
3. Optional **region head** for capped honey / capped brood if per-cell recall on caps stays low:
   semantic mask -> fill region with a hex grid at the estimated pitch -> virtual cells. Keeps
   output as cells so UI, brush and counts stay unchanged.
4. Classes: keep the 7 current ones + add `drone brood` (id 7 is already supported by
   image-splitter `countCellsAbsoluteNrs`).
5. Runtime: export to ONNX, run with onnxruntime (CPU, optional int8), Python 3.12 slim image,
   no TensorFlow. Output contract stays `[x, y, r, class, class, confidence]` for `image-splitter/src/workers/detectCells.ts`.

Licensing note: Ultralytics YOLO is AGPL-3.0. Gratheon services are open source (bee-detector is GPL),
so this is acceptable, but this repo is MIT today - the license would change. Apache-2.0 alternatives
(RT-DETR / D-FINE / RF-DETR) are a poor fit: DETR query limits (~300) are far below 3-5k cells per image.

## 4. Data plan

1. **Golden test set first**: 60-100 of our own uploaded frame photos, densely labeled, stratified:
   full frames, close-ups, narrow/queen-rearing frames, low/harsh light, honey-heavy, brood-heavy.
2. **Training set**: 300-500 of our frames, pre-annotated by the current model / first new model,
   corrected in CVAT or Label Studio (model-assisted, SAM-assisted). Tile-level labeling is fine.
3. **Public pretraining**: FAIRHiveFrames-1K (sparse labels -> train on cropped labeled regions only,
   or treat unlabeled area as ignore), DeepBee data, capped honey datasets.
4. **Feedback loop from users**: the web-app cell brush already lets users fix cells, but
   `updateRelativeCells` overwrites `files_frame_side_cells.cells` and there is no "edited" flag.
   Add `cells_model` (original prediction), `cells_edited_at`, `model_version` so user corrections
   become training labels and we can measure real-world error.
5. Augmentation: strong HSV/brightness, blur, JPEG, scale jitter around the normalized pitch, rotation.

## 5. Evaluation

Run the **current model on the golden set first** to get a baseline. Metrics:

- Cell detection precision/recall by center-distance matching (< 0.5 pitch), not IoU mAP
  (IoU mAP penalizes dense small objects, see FAIRHiveFrames discussion).
- Per-class count/percent error per frame (this is what beekeepers see).
- Per-stratum results (zoom, narrow frame, lighting, capped honey).
- Latency p50/p95 and peak RSS on the production CPU host.

Ship criteria: better than baseline on every stratum, capped honey recall >= 0.85, p95 latency and RAM
well below current.

## 6. Phases

| Phase | Work | Outcome |
|---|---|---|
| 0 | Golden set + benchmark script, baseline current model | Numbers to beat |
| 1 | Quick wins in current pipeline: no forced 6000x4000 stretch (pad to aspect), wider radius search, `MAX_CLASSIFICATION_CELLS` >= 6000 | Some relief for narrow frames/zoom while new model is built |
| 2 | Store original predictions + edited flag in image-splitter | Continuous label source |
| 3 | Dataset: public pretraining + 300-500 corrected own frames | Training data |
| 4 | Train pitch estimator + tiled YOLO detector, evaluate vs golden set | New model |
| 5 | New ONNX service behind same HTTP API, shadow mode (run both, compare), then switch | Safe rollout |
| 6 | Active learning: sample frames with user edits / low confidence for relabeling | Ongoing improvement |

## 7. Open questions

- Training compute: local Apple Silicon (MPS, slow) vs short rented GPU.
- Labeling effort owner and tool (CVAT vs Label Studio).
- License change of this repo to AGPL (if Ultralytics is used).
- Production host CPU/RAM limits to size the model (n vs s).
