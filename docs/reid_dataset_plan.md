# Re-ID Dataset Plan (Same Zebra -> Same ID)

Detection quality is now strong. Stable identity assignment now depends on identity-labeled zebra crops.

## Goal

Build a zebra identity dataset suitable for training/evaluating Re-ID embeddings.

## Minimum Viable Dataset

- Individuals: 300+
- Images per individual: 8-15 (minimum), across different times/angles
- Total crops: 5k-20k for first production-quality model

## Required Metadata Per Crop

- `image_path`
- `zebra_id` (identity label)
- `camera_id`
- `timestamp`
- `location`
- `view_side` (`left/right/unknown`)
- `quality_score` (optional but useful)

## Data Sources

1. Internal CCTV/camera-trap captures with human ID verification.
2. Conservation partner contributions (identity-confirmed tracks).
3. Public wildlife Re-ID datasets for pretraining only (domain adaptation still required).

## Split Policy for Re-ID

Use identity-aware splits:

- Train IDs and test IDs should be disjoint for strict evaluation.
- Also maintain a temporal holdout split on seen IDs for drift testing.

## Annotation Workflow

1. Run detector baseline to generate zebra crops.
2. Human reviewer assigns/validates `zebra_id`.
3. Store best-quality reference crop per ID.
4. Mark uncertain assignments with `review_required=true`.

## Evaluation Metrics

- Top-1 / Top-5 retrieval accuracy
- mAP for retrieval
- ID collision rate
- Temporal drift alerts (Hamming > 0.35) review rate

## Immediate Next Steps

1. Create `reid_manifest.csv` schema and begin logging every new validated crop.
2. Collect first 5k labeled crops with strict metadata.
3. Train/fine-tune embedding model and compare against current baseline matching behavior.
