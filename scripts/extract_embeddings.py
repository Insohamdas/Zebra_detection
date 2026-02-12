"""Generate feature embeddings from cropped zebra images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.reid import build_model


def _resolve_crop_path(raw_path: str, crops_dir: Path) -> Optional[Path]:
    candidates = [
        Path(raw_path),
        crops_dir / Path(raw_path).name,
        crops_dir / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_records(crops_dir: Path, meta_path: Optional[Path]) -> List[dict]:
    if meta_path and meta_path.exists():
        with meta_path.open() as f:
            return json.load(f)

    records = []
    for img_path in sorted(crops_dir.glob("*.jpg")):
        records.append({"crop_path": str(img_path)})
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings for zebra crops")
    parser.add_argument("--crops-dir", default="crops", type=Path, help="Directory with cropped detections")
    parser.add_argument("--meta-path", default="crops/meta.json", type=Path, help="Metadata JSON (optional)")
    parser.add_argument("--output", default="embeddings.csv", type=Path, help="CSV file to write embeddings")
    parser.add_argument("--weights", type=str, default=None, help="Optional checkpoint for the ReID model")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Projection dimensionality")
    parser.add_argument("--image-size", type=int, default=224, help="Size for square resize of crops")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps)")
    parser.add_argument("--disable-pretrained", action="store_true", help="Skip ImageNet pretrained weights")
    return parser.parse_args()


def _device_choice(force: Optional[str]) -> torch.device:
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _writer_header(embedding_dim: int) -> List[str]:
    base = [
        "crop_path",
        "source_image",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "class_id",
    ]
    base.extend([f"emb_{i}" for i in range(embedding_dim)])
    return base


def _record_row(record: dict, embedding: torch.Tensor) -> List[str]:
    bbox = record.get("bbox_xyxy", [None, None, None, None])
    return [
        record.get("crop_path"),
        record.get("source_image"),
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        record.get("confidence"),
        record.get("class_id"),
        *embedding.tolist(),
    ]


def _transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_embeddings() -> None:
    args = _parse_args()
    crops_dir = args.crops_dir
    meta_path = args.meta_path if args.meta_path else None

    crops_dir = crops_dir.resolve()
    if meta_path:
        meta_path = meta_path.resolve()

    records = _load_records(crops_dir, meta_path)
    if not records:
        raise SystemExit(f"No crops found in {crops_dir}")

    device = _device_choice(args.device)
    model = build_model(
        embedding_dim=args.embedding_dim,
        pretrained=not args.disable_pretrained,
        weights_path=args.weights,
        device=device,
    )
    model.eval()

    transform = _transform(args.image_size)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = _writer_header(args.embedding_dim)
    processed = 0

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        batch_tensors: List[torch.Tensor] = []
        batch_meta: List[dict] = []

        def _flush_batch() -> None:
            nonlocal processed
            if not batch_tensors:
                return
            batch = torch.stack(batch_tensors).to(device)
            with torch.inference_mode():
                embeddings = model(batch).cpu()
            for meta, emb in zip(batch_meta, embeddings):
                writer.writerow(_record_row(meta, emb))
            processed += len(batch_meta)
            batch_tensors.clear()
            batch_meta.clear()

        for record in records:
            crop_path = _resolve_crop_path(record.get("crop_path", ""), crops_dir)
            if crop_path is None:
                print(f"[WARN] Missing crop: {record.get('crop_path')}")
                continue

            image = Image.open(crop_path).convert("RGB")
            tensor = transform(image)
            batch_tensors.append(tensor)
            batch_meta.append(record)

            if len(batch_tensors) >= args.batch_size:
                _flush_batch()

        _flush_batch()

    print(f"Wrote {processed} embeddings to {output_path}")


if __name__ == "__main__":
    extract_embeddings()
