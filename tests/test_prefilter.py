"""Tests for lightweight frame pre-filtering."""

import cv2
import numpy as np

from zebraid.preprocessing import FramePrefilter, FramePrefilterConfig
from zebraid.pipelines.real_identify import quality_score


def _sharp_test_frame() -> np.ndarray:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    for idx in range(0, 256, 16):
        color = 255 if (idx // 16) % 2 == 0 else 40
        frame[:, idx : idx + 8] = color
    return frame


def test_prefilter_accepts_sharp_well_exposed_frame() -> None:
    prefilter = FramePrefilter()

    decision = prefilter.evaluate(_sharp_test_frame())

    assert decision.passed is True
    assert decision.reasons == ()
    assert 0.0 <= decision.score <= 1.0


def test_prefilter_rejects_blurry_frame() -> None:
    prefilter = FramePrefilter()
    blurry = cv2.GaussianBlur(_sharp_test_frame(), (51, 51), 0)

    decision = prefilter.evaluate(blurry)

    assert decision.passed is False
    assert "blurry" in decision.reasons


def test_prefilter_rejects_overexposed_frame() -> None:
    prefilter = FramePrefilter()
    overexposed = np.ones((256, 256, 3), dtype=np.uint8) * 255

    decision = prefilter.evaluate(overexposed)

    assert decision.passed is False
    assert "overexposed" in decision.reasons


def test_quality_score_uses_prefilter_score() -> None:
    ok, score = quality_score(_sharp_test_frame())

    assert ok is True
    assert 0.0 <= score <= 1.0


def test_prefilter_thresholds_are_configurable() -> None:
    config = FramePrefilterConfig(max_overexposed_fraction=0.9)
    prefilter = FramePrefilter(config=config)
    partly_bright = _sharp_test_frame()
    partly_bright[:, :64] = 255

    decision = prefilter.evaluate(partly_bright)

    assert decision.passed is True
