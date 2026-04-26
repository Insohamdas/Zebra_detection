import numpy as np
import cv2

from zebraid.experiments.evaluate import (
    EvalSample,
    PairScore,
    apply_synthetic_occlusion,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_reid_system,
    generate_paper_table,
    load_eval_samples_from_csv,
    run_default_evaluation,
)


def _sample(label: str, value: int) -> EvalSample:
    image = np.full((256, 256, 3), value, dtype=np.uint8)
    return EvalSample(zebra_label=label, image=image)


def test_apply_synthetic_occlusion_preserves_shape_and_type() -> None:
    image = np.full((256, 256, 3), 180, dtype=np.uint8)
    occluded = apply_synthetic_occlusion(image, ratio=0.35)

    assert occluded.shape == image.shape
    assert occluded.dtype == np.uint8
    assert np.any(occluded == 0)


def test_evaluate_reid_system_returns_expected_metric_keys() -> None:
    samples = [_sample(f"ZEBRA-{i:03d}", 50 + i * 10) for i in range(4)]

    def embedding_fn(image: np.ndarray) -> np.ndarray:
        mean_val = float(np.mean(image))
        vec = np.array([mean_val, 255.0 - mean_val, 0.5 * mean_val], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    metrics = evaluate_reid_system(
        samples,
        match_threshold=0.5,
        occlusion_ratio=0.35,
        embedding_fn=embedding_fn,
    )

    assert set(metrics.keys()) == {
        "accuracy",
        "false_positive_rate",
        "occlusion_accuracy",
        "roc_auc",
        "pr_auc",
    }

    for value in metrics.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_run_default_evaluation_produces_publishable_payload_shape() -> None:
    metrics = run_default_evaluation(synthetic_ids=3, match_threshold=0.5, occlusion_ratio=0.35)

    assert set(metrics.keys()) == {
        "accuracy",
        "false_positive_rate",
        "occlusion_accuracy",
        "roc_auc",
        "pr_auc",
    }
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
    assert 0.0 <= metrics["occlusion_accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_dataset_benchmark_exports_confusion_roc_pr_and_markdown(tmp_path) -> None:
    img_a = np.full((64, 64, 3), 60, dtype=np.uint8)
    img_b = np.full((64, 64, 3), 200, dtype=np.uint8)

    a_path = tmp_path / "a.jpg"
    b_path = tmp_path / "b.jpg"
    assert cv2.imwrite(str(a_path), img_a)
    assert cv2.imwrite(str(b_path), img_b)

    manifest = tmp_path / "benchmark.csv"
    manifest.write_text(
        "zebra_id,image_path\n"
        "Z1,a.jpg\n"
        "Z2,b.jpg\n",
        encoding="utf-8",
    )

    samples = load_eval_samples_from_csv(
        manifest_csv=manifest,
        image_root=tmp_path,
        label_column="zebra_id",
        image_column="image_path",
    )
    assert len(samples) == 2

    def embedding_fn(image: np.ndarray) -> np.ndarray:
        mean_val = float(np.mean(image))
        vec = np.array([mean_val, 255.0 - mean_val, 0.5 * mean_val], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    out_dir = tmp_path / "out"
    metrics = evaluate_reid_system(
        samples,
        match_threshold=0.5,
        occlusion_ratio=0.35,
        embedding_fn=embedding_fn,
        export_dir=out_dir,
        report_title="Paper Benchmark",
        dataset_name="unit-test-dataset",
    )

    assert set(metrics.keys()) == {
        "accuracy",
        "false_positive_rate",
        "occlusion_accuracy",
        "roc_auc",
        "pr_auc",
    }
    assert (out_dir / "confusion_matrix.csv").exists()
    assert (out_dir / "roc_curve.csv").exists()
    assert (out_dir / "pr_curve.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "report.md").exists()


def test_compute_roc_auc_values_in_valid_range() -> None:
    # Perfect classifier: all same-identity pairs have high similarity, all different-identity pairs have low
    perfect_scores = [
        PairScore(similarity=0.95, is_same_identity=True),
        PairScore(similarity=0.90, is_same_identity=True),
        PairScore(similarity=0.05, is_same_identity=False),
        PairScore(similarity=0.10, is_same_identity=False),
    ]
    roc_auc = compute_roc_auc(perfect_scores)
    assert isinstance(roc_auc, float)
    assert 0.0 <= roc_auc <= 1.0
    assert roc_auc > 0.7  # Should be high for perfect classifier

    # Random classifier
    random_scores = [
        PairScore(similarity=0.5, is_same_identity=True),
        PairScore(similarity=0.5, is_same_identity=False),
    ]
    roc_auc = compute_roc_auc(random_scores)
    assert 0.0 <= roc_auc <= 1.0

    # Empty list
    roc_auc = compute_roc_auc([])
    assert roc_auc == 0.0


def test_compute_pr_auc_values_in_valid_range() -> None:
    # Perfect classifier
    perfect_scores = [
        PairScore(similarity=0.95, is_same_identity=True),
        PairScore(similarity=0.90, is_same_identity=True),
        PairScore(similarity=0.05, is_same_identity=False),
        PairScore(similarity=0.10, is_same_identity=False),
    ]
    pr_auc = compute_pr_auc(perfect_scores)
    assert isinstance(pr_auc, float)
    assert 0.0 <= pr_auc <= 1.0
    assert pr_auc > 0.7  # Should be high for perfect classifier

    # Random classifier
    random_scores = [
        PairScore(similarity=0.5, is_same_identity=True),
        PairScore(similarity=0.5, is_same_identity=False),
    ]
    pr_auc = compute_pr_auc(random_scores)
    assert 0.0 <= pr_auc <= 1.0

    # Empty list
    pr_auc = compute_pr_auc([])
    assert pr_auc == 0.0


def test_generate_paper_table_has_required_columns() -> None:
    metrics = {
        "accuracy": 0.9500,
        "false_positive_rate": 0.0200,
        "occlusion_accuracy": 0.8700,
        "roc_auc": 0.9800,
        "pr_auc": 0.9600,
    }
    table = generate_paper_table(metrics)

    assert isinstance(table, str)
    assert "Accuracy" in table
    assert "ROC-AUC" in table
    assert "PR-AUC" in table
    assert "FPR" in table
    assert "Occlusion Accuracy" in table
    assert "0.9500" in table
    assert "0.9800" in table
    assert "0.9600" in table
    assert "|" in table  # Markdown table format

