"""Crop detections from frames and persist them for ReID training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image
from ultralytics import YOLO


def find_images(source: Path) -> Iterable[Path]:
    """Yield all image files under the given source path."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    if source.is_file() and source.suffix.lower() in suffixes:
        yield source
        return

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def crop_and_save_frames(
    detector_checkpoint: Path,
    source_path: Path,
    output_dir: Path,
    conf: float,
) -> list[dict]:
    """Run detection, crop boxes, and store crops on disk."""
    model = YOLO(str(detector_checkpoint))
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: list[dict] = []
    for image_path in find_images(source_path):
        results = model.predict(source=str(image_path), conf=conf, verbose=False)
        if not results:
            continue

        image = Image.open(image_path).convert("RGB")
        for result_idx, result in enumerate(results):
            boxes = getattr(result.boxes, "xyxy", None)
            if boxes is None:
                continue

            for det_idx, (xmin, ymin, xmax, ymax) in enumerate(boxes.tolist()):
                crop = image.crop((xmin, ymin, xmax, ymax))
                crop_name = f"{image_path.stem}_{result_idx}_{det_idx}.jpg"
                crop_path = output_dir / crop_name
                crop.save(crop_path, quality=95)

                meta.append(
                    {
                        "crop_path": str(crop_path.relative_to(output_dir.parent)),
                        "source_image": str(image_path),
                        "bbox_xyxy": [xmin, ymin, xmax, ymax],
                        "confidence": float(result.boxes.conf[det_idx].item()),
                        "class_id": int(result.boxes.cls[det_idx].item()),
                    }
                )

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop YOLO detections for ReID training")
    parser.add_argument(
        "--weights",
        default="runs/detect/train/weights/best.pt",
        type=Path,
        help="Path to trained YOLO checkpoint",
    )
    parser.add_argument(
        "--source",
        default="datasets/train/images",
        type=Path,
        help="Image file or directory to run the detector on",
    )
    parser.add_argument(
        "--output",
        default=Path("crops"),
        type=Path,
        help="Directory to store the cropped detections",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Confidence threshold for retaining detections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = crop_and_save_frames(args.weights, args.source, args.output, args.conf)
    meta_path = args.output / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
