# Innovation: Active Hard-Example Mining Loop

This project now includes an automated hard-example miner to push performance beyond a single training run.

## Why this is powerful

Instead of guessing which new data to label, we automatically collect difficult samples:

- false positives (model sees zebra where none/mismatch exists)
- false negatives (model misses real zebra)
- low-confidence true positives (borderline detections)

This creates a practical active-learning loop.

## Command

```bash
cd /Users/soham/Zebra_detection
.venv/bin/python -m scripts.hard_miner \
  --model /Users/soham/Zebra_detection/models/detector/zebra_v1/best.pt \
  --data-root /Users/soham/Zebra_detection/data/datasets/curated/zebra_detection/yolov8/zebra_combined_v1 \
  --split test \
  --output-dir /Users/soham/Zebra_detection/data/datasets/curated/zebra_detection/hard_mining/v1_test_mine \
  --conf-thres 0.25 \
  --iou-thres 0.5 \
  --low-conf-tp-thres 0.45
```

## Output structure

```text
hard_mining/v1_test_mine/
├── false_positive/
│   ├── images/
│   └── labels/
├── false_negative/
│   ├── images/
│   └── labels/
├── low_conf_true_positive/
│   ├── images/
│   └── labels/
└── summary.txt
```

## Recommended loop

1. Run miner after each validation.
2. Relabel/clean the mined samples.
3. Build next dataset version with mined samples:

```bash
cd /Users/soham/Zebra_detection
.venv/bin/python -m scripts.build_next_dataset \
  --base-dataset /Users/soham/Zebra_detection/data/datasets/curated/zebra_detection/yolov8/zebra_combined_v1 \
  --hard-mining-root /Users/soham/Zebra_detection/data/datasets/curated/zebra_detection/hard_mining/v1_test_mine \
  --output-dataset /Users/soham/Zebra_detection/data/datasets/curated/zebra_detection/yolov8/zebra_combined_v2
```

4. Retrain from previous `best.pt`.
5. Compare test metrics against baseline.
