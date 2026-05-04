"""Hard-example miner for YOLO detection datasets.

Runs inference on a split and exports difficult samples for relabel/retraining:
- false positives (predictions with no matching GT)
- false negatives (GT with no matching prediction)
- low-confidence true positives
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Box:
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard examples from YOLO dataset split")
    parser.add_argument("--model", required=True, help="Path to detector weights (.pt)")
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/valid/test")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--output-dir", required=True, help="Output root for mined samples")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Inference confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument(
        "--low-conf-tp-thres",
        type=float,
        default=0.45,
        help="TP confidence below this is considered hard",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="Optional max images to process")
    return parser.parse_args()


def yolo_to_xyxy(line: str, w: int, h: int) -> Box | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls_id = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:])
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return Box(cls_id=cls_id, conf=1.0, x1=x1, y1=y1, x2=x2, y2=y2)


def iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_gt_boxes(label_path: Path, w: int, h: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text().splitlines():
        box = yolo_to_xyxy(line, w, h)
        if box is not None:
            boxes.append(box)
    return boxes


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "false_positive": base / "false_positive",
        "false_negative": base / "false_negative",
        "low_conf_true_positive": base / "low_conf_true_positive",
    }
    for path in dirs.values():
        (path / "images").mkdir(parents=True, exist_ok=True)
        (path / "labels").mkdir(parents=True, exist_ok=True)
    return dirs


def copy_pair(image_path: Path, label_path: Path, target_group: Path) -> None:
    shutil.copy2(image_path, target_group / "images" / image_path.name)
    if label_path.exists():
        shutil.copy2(label_path, target_group / "labels" / label_path.name)


def run() -> None:
    args = parse_args()
    model = YOLO(args.model)

    data_root = Path(args.data_root)
    image_dir = data_root / args.split / "images"
    label_dir = data_root / args.split / "labels"
    output_root = Path(args.output_dir)
    dirs = ensure_dirs(output_root)

    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    stats = {
        "images": 0,
        "false_positive": 0,
        "false_negative": 0,
        "low_conf_true_positive": 0,
    }

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        gt_boxes = load_gt_boxes(label_path, w, h)

        result = model.predict(
            source=str(image_path),
            conf=args.conf_thres,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        pred_boxes: list[Box] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                pred_boxes.append(
                    Box(
                        cls_id=int(b.cls[0].item()),
                        conf=float(b.conf[0].item()),
                        x1=float(b.xyxy[0][0].item()),
                        y1=float(b.xyxy[0][1].item()),
                        x2=float(b.xyxy[0][2].item()),
                        y2=float(b.xyxy[0][3].item()),
                    )
                )

        stats["images"] += 1
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        for pi, pb in enumerate(pred_boxes):
            best_idx = -1
            best_iou = 0.0
            for gi, gb in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                if pb.cls_id != gb.cls_id:
                    continue
                ov = iou(pb, gb)
                if ov > best_iou:
                    best_iou = ov
                    best_idx = gi
            if best_idx >= 0 and best_iou >= args.iou_thres:
                gt_matched[best_idx] = True
                pred_matched[pi] = True
                if pb.conf < args.low_conf_tp_thres:
                    copy_pair(image_path, label_path, dirs["low_conf_true_positive"])
                    stats["low_conf_true_positive"] += 1
                    break

        has_fp = any(not m for m in pred_matched)
        has_fn = any(not m for m in gt_matched)
        if has_fp:
            copy_pair(image_path, label_path, dirs["false_positive"])
            stats["false_positive"] += 1
        if has_fn:
            copy_pair(image_path, label_path, dirs["false_negative"])
            stats["false_negative"] += 1

    report_path = output_root / "summary.txt"
    report_path.write_text(
        "\n".join(
            [
                f"images_processed={stats['images']}",
                f"false_positive_images={stats['false_positive']}",
                f"false_negative_images={stats['false_negative']}",
                f"low_conf_true_positive_images={stats['low_conf_true_positive']}",
            ]
        )
        + "\n"
    )
    print(report_path.read_text().strip())


if __name__ == "__main__":
    run()
