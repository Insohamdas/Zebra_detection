"""Evaluation utilities for paper-ready zebra ReID metrics.

Provides three publishable metrics:
1. accuracy: same-zebra pair should resolve to same ID
2. false_positive_rate: different-zebra pair incorrectly resolves to same ID
3. occlusion_accuracy: occluded query still resolves to original ID
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from zebraid.feature_engine import FeatureEncoder
from zebraid.preprocessing import ZebraSegmenter, prepare_tensor


@dataclass(frozen=True, slots=True)
class EvalSample:
    """A single labeled sample used for metric computation."""

    zebra_label: str
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class PairScore:
    """Pairwise similarity score used for ROC/PR generation."""

    similarity: float
    is_same_identity: bool


def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be grayscale/BGR/BGRA")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(image.max(initial=0.0)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def apply_synthetic_occlusion(image: np.ndarray, *, ratio: float = 0.35) -> np.ndarray:
    """Apply a deterministic rectangular occlusion to simulate partial visibility."""

    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be in (0, 1)")

    occluded = _ensure_bgr_uint8(image).copy()
    h, w = occluded.shape[:2]
    occ_w = max(1, int(w * ratio))
    x1 = w // 2 - occ_w // 2
    x2 = min(w, x1 + occ_w)
    occluded[:, x1:x2] = 0
    return occluded


def _embedding_for_image(
    image: np.ndarray,
    *,
    encoder: FeatureEncoder,
    segmenter: ZebraSegmenter,
) -> np.ndarray:
    frame_tensor = prepare_tensor(image, segmenter=segmenter)
    with torch.no_grad():
        embedding = encoder.encode(frame_tensor)
    emb = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _nearest_match(
    embedding: np.ndarray,
    *,
    registry_embeddings: list[np.ndarray],
    registry_ids: list[str],
    threshold: float,
) -> str:
    """Return existing ID if nearest neighbor is within threshold, else create a new ID."""

    if not registry_embeddings:
        new_id = f"EVAL-{len(registry_ids):05d}"
        registry_embeddings.append(embedding)
        registry_ids.append(new_id)
        return new_id

    dists = [float(np.linalg.norm(embedding - ref)) for ref in registry_embeddings]
    best_idx = int(np.argmin(dists))
    best_dist = dists[best_idx]

    if best_dist < threshold:
        return registry_ids[best_idx]

    new_id = f"EVAL-{len(registry_ids):05d}"
    registry_embeddings.append(embedding)
    registry_ids.append(new_id)
    return new_id


def _nearest_label(
    query_embedding: np.ndarray,
    gallery_embeddings: Sequence[np.ndarray],
    gallery_labels: Sequence[str],
) -> str:
    if not gallery_embeddings:
        raise ValueError("gallery_embeddings must not be empty")

    dists = [float(np.linalg.norm(query_embedding - emb)) for emb in gallery_embeddings]
    best_idx = int(np.argmin(dists))
    return str(gallery_labels[best_idx])


def _distance_to_similarity(distance: float) -> float:
    """Convert L2 distance between normalized vectors to bounded similarity [0, 1]."""

    sim = 1.0 - min(max(distance, 0.0), 2.0) / 2.0
    return float(np.clip(sim, 0.0, 1.0))


def _pair_scores(embeddings: Sequence[np.ndarray], labels: Sequence[str]) -> list[PairScore]:
    scores: list[PairScore] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
            scores.append(
                PairScore(
                    similarity=_distance_to_similarity(dist),
                    is_same_identity=labels[i] == labels[j],
                )
            )
    return scores


def _compute_roc_pr(points: Sequence[PairScore]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Compute ROC and PR curve points from pair scores."""

    if not points:
        return [], []

    thresholds = sorted({p.similarity for p in points}, reverse=True)
    thresholds = [1.01, *thresholds, -0.01]

    roc: list[dict[str, float]] = []
    pr: list[dict[str, float]] = []

    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for p in points:
            pred_same = p.similarity >= threshold
            if pred_same and p.is_same_identity:
                tp += 1
            elif pred_same and not p.is_same_identity:
                fp += 1
            elif (not pred_same) and p.is_same_identity:
                fn += 1
            else:
                tn += 1

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tpr

        roc.append(
            {
                "threshold": round(float(threshold), 6),
                "tpr": round(float(tpr), 6),
                "fpr": round(float(fpr), 6),
            }
        )
        pr.append(
            {
                "threshold": round(float(threshold), 6),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
            }
        )

    return roc, pr


def compute_roc_auc(points: Sequence[PairScore]) -> float:
    """Compute ROC-AUC from pair similarity scores."""

    if not points:
        return 0.0

    # Sort by similarity (descending)
    sorted_points = sorted(points, key=lambda p: p.similarity, reverse=True)

    # Count total positives and negatives
    total_pos = sum(1 for p in points if p.is_same_identity)
    total_neg = sum(1 for p in points if not p.is_same_identity)

    if total_pos == 0 or total_neg == 0:
        return 0.0

    # Accumulate TP and FP as we move down the ranking
    tp = 0
    fp = 0
    auc = 0.0
    prev_tpr = 0.0

    for p in sorted_points:
        if p.is_same_identity:
            tp += 1
        else:
            fp += 1

        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - (fp - 1) / total_neg) * tpr if fp > 0 else 0.0

    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(points: Sequence[PairScore]) -> float:
    """Compute PR-AUC (area under precision-recall curve) from pair scores."""

    if not points:
        return 0.0

    # Sort by similarity (descending)
    sorted_points = sorted(points, key=lambda p: p.similarity, reverse=True)

    total_pos = sum(1 for p in points if p.is_same_identity)

    if total_pos == 0:
        return 0.0

    tp = 0
    auc = 0.0
    prev_recall = 0.0

    for p in sorted_points:
        if p.is_same_identity:
            tp += 1

        retrieved = sorted_points.index(p) + 1
        precision = tp / retrieved if retrieved > 0 else 0.0
        recall = tp / total_pos

        auc += (recall - prev_recall) * precision
        prev_recall = recall

    return float(np.clip(auc, 0.0, 1.0))


def _confusion_matrix(
    *,
    actual_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> tuple[list[str], list[list[int]]]:
    labels = sorted(set(actual_labels) | set(predicted_labels))
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for actual, predicted in zip(actual_labels, predicted_labels, strict=True):
        matrix[index[actual]][index[predicted]] += 1

    return labels, matrix


def _write_confusion_matrix_csv(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _write_curve_csv(path: Path, rows: Sequence[dict[str, float]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0.0) for key in columns})


def generate_paper_table(metrics: dict[str, float]) -> str:
    """Generate a compact markdown table for direct manuscript use.
    
    Returns:
        Markdown table string with key ReID metrics suitable for paper tables.
    """
    lines: list[str] = []
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Accuracy | {metrics.get('accuracy', 0.0):.4f} |")
    if "roc_auc" in metrics:
        lines.append(f"| ROC-AUC | {metrics['roc_auc']:.4f} |")
    if "pr_auc" in metrics:
        lines.append(f"| PR-AUC | {metrics['pr_auc']:.4f} |")
    lines.append(f"| FPR | {metrics.get('false_positive_rate', 0.0):.4f} |")
    lines.append(f"| Occlusion Accuracy | {metrics.get('occlusion_accuracy', 0.0):.4f} |")
    return "\n".join(lines)


def _write_markdown_report(
    *,
    path: Path,
    title: str,
    dataset_name: str,
    metrics: dict[str, float],
    confusion_labels: Sequence[str],
    confusion_matrix: Sequence[Sequence[int]],
    roc_csv: Path,
    pr_csv: Path,
    confusion_csv: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Dataset: `{dataset_name}`")
    lines.append("")
    lines.append("## Paper Results Table")
    lines.append("")
    lines.append(generate_paper_table(metrics))
    lines.append("")
    lines.append("## Detailed Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| accuracy | {metrics['accuracy']:.4f} |")
    lines.append(f"| false_positive_rate | {metrics['false_positive_rate']:.4f} |")
    lines.append(f"| occlusion_accuracy | {metrics['occlusion_accuracy']:.4f} |")
    if "roc_auc" in metrics:
        lines.append(f"| roc_auc | {metrics['roc_auc']:.4f} |")
    if "pr_auc" in metrics:
        lines.append(f"| pr_auc | {metrics['pr_auc']:.4f} |")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("| actual\\predicted | " + " | ".join(confusion_labels) + " |")
    lines.append("|---|" + "---|" * len(confusion_labels))
    for label, row in zip(confusion_labels, confusion_matrix, strict=True):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    lines.append("## Exports")
    lines.append("")
    lines.append(f"- ROC CSV: `{roc_csv.name}`")
    lines.append(f"- PR CSV: `{pr_csv.name}`")
    lines.append(f"- Confusion matrix CSV: `{confusion_csv.name}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_eval_samples_from_csv(
    *,
    manifest_csv: str | Path,
    image_root: str | Path | None = None,
    label_column: str = "zebra_id",
    image_column: str = "image_path",
) -> list[EvalSample]:
    """Load labeled evaluation samples from a CSV manifest.

    Expected columns:
    - label column (default: zebra_id)
    - image path column (default: image_path)
    """

    manifest_path = Path(manifest_csv)
    root = Path(image_root) if image_root is not None else manifest_path.parent

    samples: list[EvalSample] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {manifest_path}")

        if image_column not in reader.fieldnames and image_column == "image_path" and "image_id" in reader.fieldnames:
            image_column = "image_id"

        if label_column not in reader.fieldnames:
            raise ValueError(f"label column '{label_column}' not found in {manifest_path}")
        if image_column not in reader.fieldnames:
            raise ValueError(f"image column '{image_column}' not found in {manifest_path}")

        for row in reader:
            label = str(row[label_column]).strip()
            image_ref = str(row[image_column]).strip()
            if not label or not image_ref:
                continue

            image_path = Path(image_ref)
            if not image_path.is_absolute():
                image_path = root / image_path

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"unable to load image from manifest: {image_path}")

            samples.append(EvalSample(zebra_label=label, image=image))

    if not samples:
        raise ValueError("manifest did not yield any valid evaluation samples")

    return samples


def _build_eval_samples(synthetic_ids: int = 6) -> list[EvalSample]:
    """Generate deterministic synthetic zebra-like samples for reproducible evaluation."""

    rng = np.random.default_rng(42)
    samples: list[EvalSample] = []

    for idx in range(synthetic_ids):
        label = f"ZEBRA-{idx:03d}"
        base = np.zeros((256, 256, 3), dtype=np.uint8)

        # Zebra-like stripe signature that differs by label.
        stripe_spacing = 12 + idx
        stripe_thickness = 4 + (idx % 3)
        for x in range(0, 256, stripe_spacing):
            cv2.line(base, (x, 0), (x, 255), (255, 255, 255), stripe_thickness)

        # Add deterministic but unique texture/noise per ID.
        noise = rng.integers(0, 18, size=base.shape, dtype=np.uint8)
        image = cv2.add(base, noise)

        # Slightly different color tint per ID.
        tint = np.array([idx * 3 % 25, idx * 5 % 25, idx * 7 % 25], dtype=np.uint8)
        image = cv2.add(image, np.tile(tint, (256, 256, 1)))

        samples.append(EvalSample(zebra_label=label, image=image))

    return samples


def evaluate_reid_system(
    samples: Sequence[EvalSample],
    *,
    match_threshold: float = 0.5,
    occlusion_ratio: float = 0.35,
    embedding_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    export_dir: str | Path | None = None,
    report_title: str = "ZEBRAID Benchmark Report",
    dataset_name: str = "in-memory samples",
) -> dict[str, float]:
    """Compute core paper metrics for zebra re-identification."""

    if not samples:
        raise ValueError("samples must not be empty")

    if embedding_fn is None:
        encoder = FeatureEncoder()
        segmenter = ZebraSegmenter(backend="otsu")

        def embedding_fn(image: np.ndarray) -> np.ndarray:
            return _embedding_for_image(image, encoder=encoder, segmenter=segmenter)

    labels = [sample.zebra_label for sample in samples]
    embeddings = [embedding_fn(sample.image) for sample in samples]

    # 1) Accuracy via leave-one-out nearest-label prediction
    predicted_labels: list[str] = []
    for i, emb in enumerate(embeddings):
        if len(embeddings) == 1:
            predicted_labels.append(labels[i])
            continue

        gallery_emb = [embeddings[j] for j in range(len(embeddings)) if j != i]
        gallery_labels = [labels[j] for j in range(len(labels)) if j != i]
        predicted_labels.append(_nearest_label(emb, gallery_emb, gallery_labels))

    same_success = sum(1 for actual, pred in zip(labels, predicted_labels, strict=True) if actual == pred)
    same_total = len(labels)

    # 2) False positive rate from different-identity pairs under matching threshold
    pair_scores = _pair_scores(embeddings, labels)
    diff_pairs = [p for p in pair_scores if not p.is_same_identity]
    diff_errors = sum(
        1
        for p in diff_pairs
        if (1.0 - p.similarity) * 2.0 < match_threshold
    )
    diff_total = len(diff_pairs)

    # 3) Occlusion accuracy
    occluded_embeddings = [embedding_fn(apply_synthetic_occlusion(sample.image, ratio=occlusion_ratio)) for sample in samples]
    occ_success = 0
    occ_total = len(samples)
    for i, occ_emb in enumerate(occluded_embeddings):
        pred_label = _nearest_label(occ_emb, embeddings, labels)
        if pred_label == labels[i]:
            occ_success += 1

    accuracy = float(same_success) / float(same_total) if same_total else 0.0
    false_positive_rate = float(diff_errors) / float(diff_total) if diff_total else 0.0
    occlusion_accuracy = float(occ_success) / float(occ_total) if occ_total else 0.0
    roc_auc = compute_roc_auc(pair_scores)
    pr_auc = compute_pr_auc(pair_scores)

    confusion_labels, confusion_matrix = _confusion_matrix(
        actual_labels=labels,
        predicted_labels=predicted_labels,
    )
    roc_curve, pr_curve = _compute_roc_pr(pair_scores)

    metrics = {
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "occlusion_accuracy": round(occlusion_accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }

    if export_dir is not None:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        confusion_csv = export_path / "confusion_matrix.csv"
        roc_csv = export_path / "roc_curve.csv"
        pr_csv = export_path / "pr_curve.csv"
        report_md = export_path / "report.md"
        metrics_json = export_path / "metrics.json"

        _write_confusion_matrix_csv(confusion_csv, confusion_labels, confusion_matrix)
        _write_curve_csv(roc_csv, roc_curve, columns=("threshold", "tpr", "fpr"))
        _write_curve_csv(pr_csv, pr_curve, columns=("threshold", "precision", "recall"))
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_markdown_report(
            path=report_md,
            title=report_title,
            dataset_name=dataset_name,
            metrics=metrics,
            confusion_labels=confusion_labels,
            confusion_matrix=confusion_matrix,
            roc_csv=roc_csv,
            pr_csv=pr_csv,
            confusion_csv=confusion_csv,
        )

    return metrics


def run_default_evaluation(
    *,
    synthetic_ids: int = 6,
    match_threshold: float = 0.5,
    occlusion_ratio: float = 0.35,
) -> dict[str, float]:
    """Run evaluation on deterministic synthetic zebra-like samples."""

    samples = _build_eval_samples(synthetic_ids=synthetic_ids)
    return evaluate_reid_system(
        samples,
        match_threshold=match_threshold,
        occlusion_ratio=occlusion_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate zebra ReID metrics for experiments/papers")
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Optional CSV manifest for dataset-driven benchmark mode",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory for relative image paths in manifest CSV",
    )
    parser.add_argument(
        "--label-column",
        default="zebra_id",
        help="Column name containing ground-truth zebra identity labels",
    )
    parser.add_argument(
        "--image-column",
        default="image_path",
        help="Column name containing image paths (or image_id)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/outputs",
        help="Directory to export confusion matrix, ROC/PR CSVs, and markdown report",
    )
    parser.add_argument(
        "--report-title",
        default="ZEBRAID Benchmark Report",
        help="Title used in generated markdown report",
    )
    parser.add_argument(
        "--synthetic-ids",
        type=int,
        default=6,
        help="Number of synthetic zebra identities used for quick evaluation",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="L2 distance threshold used by matching engine",
    )
    parser.add_argument(
        "--occlusion-ratio",
        type=float,
        default=0.35,
        help="Occluded width ratio used for robustness test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.manifest_csv:
        samples = load_eval_samples_from_csv(
            manifest_csv=args.manifest_csv,
            image_root=args.image_root,
            label_column=args.label_column,
            image_column=args.image_column,
        )
        metrics = evaluate_reid_system(
            samples,
            match_threshold=args.match_threshold,
            occlusion_ratio=args.occlusion_ratio,
            export_dir=args.output_dir,
            report_title=args.report_title,
            dataset_name=str(args.manifest_csv),
        )
    else:
        metrics = run_default_evaluation(
            synthetic_ids=args.synthetic_ids,
            match_threshold=args.match_threshold,
            occlusion_ratio=args.occlusion_ratio,
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
