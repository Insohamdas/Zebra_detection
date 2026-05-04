# Detection Data Improvement Plan

This plan targets better detector robustness without changing the current architecture.

## Goal

Improve generalization on hard field conditions while keeping high precision.

## Priority Data Additions

1. Hard negatives
- Non-zebra striped patterns (fences, shadows, clothing, painted objects)
- Empty frames from camera traps and CCTV scenes

2. Hard positives
- Distant zebras (small box sizes)
- Partial zebras (occlusion by bushes/herd overlap)
- Motion blur, low light, rain, dust
- Different camera angles and focal lengths

3. Domain coverage
- Mix conservation photos + CCTV + camera-trap stills
- At least 3+ geographic locations and 3+ camera setups

## Labeling Rules

1. Draw full-body visible extent where possible.
2. For severe truncation, annotate only visible zebra body region.
3. Ignore extremely tiny boxes below practical inference size threshold unless needed for product use.
4. Keep one class only: `zebra`.

## Split Policy (Important)

Use leakage-safe split by source group, not random frame-level split:

- Group key: `camera_id + date_block + location`
- Target ratio: train 80 / val 10 / test 10
- Never place near-duplicate frames across splits.

## Quality Gate Before Training

1. Validate 1:1 image-label pairing.
2. Validate class IDs are only `0`.
3. Validate normalized box coordinates in `[0,1]`.
4. Remove broken/corrupt images.

## Next Retrain Recipe

1. Add 20-30% hard examples to dataset.
2. Retrain from current baseline `best.pt` (fine-tune).
3. Early stop near best val mAP50-95 epoch.
4. Re-run held-out test split report and compare against baseline v1.
