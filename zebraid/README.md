# ZEBRAID package

This package is the modular home for the zebra detection and re-identification pipeline.

## Phase 0 goals

- lock the scope to one species and one view
- keep the stack explicit and small
- make every module independently testable
- add a minimal API surface before moving pipeline logic

## Phase 1 focus

- define a strict dataset manifest schema for offline data
- ingest camera-trap and public dataset exports for bootstrapping
- load, resize, and normalize images consistently
- reject blurry or low-light frames before training
- connect a CCTV stream directly for live identification and new-ID generation

## Module map

- `data/` - dataset and asset handling
- `pipelines/` - live CCTV orchestration
- `preprocessing/` - crop extraction and image normalization
- `segmentation/` - mask and region logic
- `feature_engine/` - embedding extraction and transforms
- `id_generator/` - identity creation
- `registry/` - metadata and persistence
- `matching/` - similarity search and decisions
- `api/` - FastAPI app and routers
- `experiments/` - exploratory work and evaluation notebooks

## Migration plan

The legacy pipeline can be moved into this structure one module at a time without breaking the whole project.
