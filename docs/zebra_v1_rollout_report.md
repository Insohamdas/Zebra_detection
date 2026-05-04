# Zebra v1 Rollout Report

Date: 2026-05-04

## 1) Deployment Baseline

- Active detector baseline packaged at:
  - `/Users/soham/Zebra_detection/models/detector/zebra_v1/best.pt`
- API now defaults to this trained detector path, with override support via:
  - `DETECTOR_MODEL_PATH`

## 2) Held-out Validation (Test Split)

Evaluated with:

```bash
.venv/bin/yolo detect val \
  model=/Users/soham/Zebra_detection/data/runs/zebra_combined_v1/weights/best.pt \
  data=/Users/soham/Zebra_detection/data/datasets/curated/zebra_detection/yolov8/zebra_combined_v1/data.yaml \
  split=test imgsz=640
```

Results:

- Images: `172`
- Instances: `283`
- Precision: `0.963`
- Recall: `0.918`
- mAP50: `0.976`
- mAP50-95: `0.816`

Artifacts:
- `/Users/soham/Zebra_detection/data/runs_eval/zebra_combined_v1_test`

## 3) Baseline Freeze

Versioned detector package created:

- `/Users/soham/Zebra_detection/models/detector/zebra_v1/README.md`
- `/Users/soham/Zebra_detection/models/detector/zebra_v1/best.pt`
- `/Users/soham/Zebra_detection/models/detector/zebra_v1/train_args.yaml`
- `/Users/soham/Zebra_detection/models/detector/zebra_v1/train_results.csv`

## 4) Detection Data Quality Plan

Created:

- `/Users/soham/Zebra_detection/docs/detection_data_improvement_plan.md`

Focus:
- hard negatives
- hard positives (occlusion/blur/low-light)
- leakage-safe split policy
- strict annotation and QA gates

## 5) Re-ID Dataset Plan

Created:

- `/Users/soham/Zebra_detection/docs/reid_dataset_plan.md`

Focus:
- identity-labeled zebra crops
- metadata schema
- identity-aware splits
- retrieval and drift metrics
