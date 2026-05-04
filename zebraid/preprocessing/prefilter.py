"""Fast frame pre-filtering before expensive zebra identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FramePrefilterConfig:
    """Thresholds for rejecting frames before the main identification pipeline."""

    min_blur_variance: float = 30.0
    min_side: int = 64
    overexposed_pixel_threshold: int = 245
    max_overexposed_fraction: float = 0.5
    model_reject_threshold: float = 0.8


@dataclass(frozen=True, slots=True)
class FramePrefilterDecision:
    """Result from the lightweight frame pre-filter."""

    passed: bool
    score: float
    reasons: tuple[str, ...]
    blur_variance: float
    overexposed_fraction: float
    model_bad_probability: float | None = None


class ResNet18FramePrefilter:
    """Optional ResNet-18 bad-frame classifier.

    This class expects a project-trained checkpoint. The checkpoint should output
    either two logits ``[bad, good]`` or one logit where sigmoid(logit) is the
    probability of a bad frame. No ImageNet weights are downloaded here.
    """

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def bad_probability(self, frame: np.ndarray) -> float:
        """Return the model probability that a frame is bad."""

        torch = self.torch
        frame_bgr = _coerce_bgr_uint8(frame)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        if logits.shape[1] == 1:
            return float(torch.sigmoid(logits[0, 0]).detach().cpu())

        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 0].detach().cpu())


class FramePrefilter:
    """Reject obviously unusable frames before detection, segmentation, and encoding."""

    def __init__(
        self,
        config: FramePrefilterConfig | None = None,
        model: ResNet18FramePrefilter | None = None,
    ) -> None:
        self.config = config or FramePrefilterConfig()
        self.model = model

    def evaluate(self, frame: np.ndarray) -> FramePrefilterDecision:
        """Evaluate a frame using heuristic checks and an optional ResNet-18 model."""

        frame_bgr = _coerce_bgr_uint8(frame)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposed_fraction = float(np.mean(gray >= self.config.overexposed_pixel_threshold))

        reasons: list[str] = []
        if height < self.config.min_side or width < self.config.min_side:
            reasons.append("too_small")
        if blur_variance < self.config.min_blur_variance:
            reasons.append("blurry")
        if overexposed_fraction > self.config.max_overexposed_fraction:
            reasons.append("overexposed")

        model_bad_probability = None
        if self.model is not None:
            model_bad_probability = self.model.bad_probability(frame_bgr)
            if model_bad_probability >= self.config.model_reject_threshold:
                reasons.append("model_rejected")

        blur_score = min(1.0, blur_variance / max(self.config.min_blur_variance, 1.0))
        exposure_score = max(0.0, 1.0 - (overexposed_fraction / max(self.config.max_overexposed_fraction, 1e-6)))
        score = float(np.clip((blur_score + exposure_score) / 2.0, 0.0, 1.0))

        return FramePrefilterDecision(
            passed=not reasons,
            score=score,
            reasons=tuple(reasons),
            blur_variance=blur_variance,
            overexposed_fraction=overexposed_fraction,
            model_bad_probability=model_bad_probability,
        )


def _coerce_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert common image layouts to a BGR uint8 frame."""

    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be grayscale, BGR, or BGRA")

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and float(frame.max(initial=0.0)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame
