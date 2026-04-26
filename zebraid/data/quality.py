"""Visual quality filtering for zebra image acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class QualityFilterConfig:
    """Thresholds used to reject unusable images."""

    min_record_quality_score: float = 0.5
    min_visual_quality_score: float = 0.5
    min_blur_variance: float = 80.0
    min_brightness: float = 35.0
    min_contrast: float = 15.0


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Computed image quality signals."""

    blur_variance: float
    brightness: float
    contrast: float
    visual_quality_score: float
    is_blurry: bool
    is_low_light: bool
    is_low_contrast: bool


@dataclass(frozen=True, slots=True)
class QualityDecision:
    """Final decision that combines image quality and record quality."""

    record_quality_score: float
    visual_quality_score: float
    combined_quality_score: float
    passed: bool
    reasons: tuple[str, ...]


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError("image must be a 2D grayscale or 3D color array")

    gray = gray.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
        gray = gray * 255.0
    return gray


def _clip_unit_interval(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def assess_quality(image: np.ndarray, *, config: QualityFilterConfig | None = None) -> QualityMetrics:
    """Compute blur, brightness, contrast, and a visual quality score."""

    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")

    config = config or QualityFilterConfig()
    gray = _as_grayscale(image)

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    sharpness_score = 1.0 - float(np.exp(-blur_variance / max(config.min_blur_variance, 1.0)))
    brightness_score = 1.0 - min(1.0, abs(brightness - 127.5) / 127.5)
    contrast_score = min(1.0, contrast / max(config.min_contrast * 4.0, 1.0))
    visual_quality_score = _clip_unit_interval(
        0.5 * sharpness_score + 0.25 * brightness_score + 0.25 * contrast_score
    )

    return QualityMetrics(
        blur_variance=blur_variance,
        brightness=brightness,
        contrast=contrast,
        visual_quality_score=visual_quality_score,
        is_blurry=blur_variance < config.min_blur_variance,
        is_low_light=brightness < config.min_brightness,
        is_low_contrast=contrast < config.min_contrast,
    )


def evaluate_quality(
    image: np.ndarray,
    *,
    record_quality_score: float = 1.0,
    config: QualityFilterConfig | None = None,
) -> QualityDecision:
    """Combine source metadata quality with runtime image quality."""

    config = config or QualityFilterConfig()
    metrics = assess_quality(image, config=config)

    try:
        record_quality = float(record_quality_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("record_quality_score must be numeric") from exc

    combined_quality_score = _clip_unit_interval(
        (record_quality + metrics.visual_quality_score) / 2.0
    )

    reasons: list[str] = []
    if record_quality < config.min_record_quality_score:
        reasons.append("record_quality_below_threshold")
    if metrics.is_blurry:
        reasons.append("image_blurry")
    if metrics.is_low_light:
        reasons.append("low_light_frame")
    if metrics.is_low_contrast:
        reasons.append("low_contrast_frame")
    if metrics.visual_quality_score < config.min_visual_quality_score:
        reasons.append("visual_quality_below_threshold")

    return QualityDecision(
        record_quality_score=record_quality,
        visual_quality_score=metrics.visual_quality_score,
        combined_quality_score=combined_quality_score,
        passed=not reasons,
        reasons=tuple(reasons),
    )