from pathlib import Path

import cv2
import numpy as np

from zebraid.data import (
    QualityFilterConfig,
    ZebraDataLoader,
    ZebraDataRecord,
    build_path_resolver,
    evaluate_quality,
    load_manifest,
    load_records_from_csv,
    save_manifest,
)


def _make_test_image() -> np.ndarray:
    image = np.zeros((96, 144, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 16), (126, 80), (255, 255, 255), thickness=-1)
    cv2.line(image, (0, 95), (143, 0), (0, 0, 255), thickness=3)
    return image


def test_record_validation_and_roundtrip() -> None:
    record = ZebraDataRecord.from_mapping(
        {
            "image_id": "IMG_001",
            "gps": "-2.345,34.123",
            "timestamp": "2026-04-22T12:34:56Z",
            "side": "LEFT",
            "quality_score": 0.91,
        }
    )

    assert record.image_id == "IMG_001"
    assert record.gps == "-2.345,34.123"
    assert record.timestamp == "2026-04-22T12:34:56Z"
    assert record.side == "left"
    assert record.quality_score == 0.91
    assert record.to_mapping() == {
        "image_id": "IMG_001",
        "gps": "-2.345,34.123",
        "timestamp": "2026-04-22T12:34:56Z",
        "side": "left",
        "quality_score": 0.91,
    }


def test_quality_filter_rejects_blurry_and_dark_images() -> None:
    config = QualityFilterConfig(
        min_record_quality_score=0.5,
        min_visual_quality_score=0.45,
        min_blur_variance=40.0,
        min_brightness=20.0,
        min_contrast=8.0,
    )

    sharp = _make_test_image()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    dark = np.zeros_like(sharp)

    sharp_decision = evaluate_quality(sharp, record_quality_score=0.9, config=config)
    blurred_decision = evaluate_quality(blurred, record_quality_score=0.9, config=config)
    dark_decision = evaluate_quality(dark, record_quality_score=0.9, config=config)

    assert sharp_decision.passed is True
    assert blurred_decision.passed is False
    assert "image_blurry" in blurred_decision.reasons
    assert dark_decision.passed is False
    assert "low_light_frame" in dark_decision.reasons


def test_loader_resizes_and_normalizes(tmp_path: Path) -> None:
    image = _make_test_image()
    image_path = tmp_path / "IMG_002.jpg"
    assert cv2.imwrite(str(image_path), image)

    record = ZebraDataRecord(
        image_id="IMG_002",
        gps="-2.345,34.123",
        timestamp="2026-04-22T12:34:56Z",
        side="right",
        quality_score=0.95,
    )

    loader = ZebraDataLoader(
        [record],
        lambda _: image_path,
        image_size=(512, 512),
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.3,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    samples = loader.load_all()

    assert len(samples) == 1
    sample = samples[0]
    assert sample.image_path == image_path
    assert sample.image.shape == (512, 512, 3)
    assert sample.image.dtype == np.float32
    assert 0.0 <= float(sample.image.min()) <= 1.0
    assert 0.0 <= float(sample.image.max()) <= 1.0


def test_manifest_csv_and_jsonl_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "image_id,gps,timestamp,side,quality_score\n"
        'IMG_003,"-2.0,34.0",2026-04-22T12:34:56+00:00,left,0.8\n',
        encoding="utf-8",
    )

    records_from_csv = load_records_from_csv(csv_path)
    assert len(records_from_csv) == 1
    assert records_from_csv[0].image_id == "IMG_003"

    jsonl_path = tmp_path / "manifest.jsonl"
    save_manifest(records_from_csv, jsonl_path)
    records_from_jsonl = load_manifest(jsonl_path)
    assert records_from_jsonl == records_from_csv


def test_build_path_resolver_finds_jpeg(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "IMG_004.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    resolver = build_path_resolver(image_root)
    record = ZebraDataRecord(
        image_id="IMG_004",
        gps="-2.0,34.0",
        timestamp="2026-04-22T12:34:56+00:00",
        side="left",
        quality_score=0.7,
    )

    assert resolver(record) == image_path