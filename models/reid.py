"""Lightweight ResNet50-based backbone for zebra ReID embeddings."""

from __future__ import annotations

import shutil
import ssl
import warnings
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

try:  # Prefer certifi for consistent CA bundles
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


class ReIDModel(nn.Module):
    """ResNet50 backbone with a projection head for L2-normalized embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        if weights is None:
            backbone = models.resnet50(weights=None)
        else:
            backbone = self._build_with_cached_weights(weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        proj = self.embedding(feats)
        return F.normalize(proj, dim=1)

    def load_checkpoint(self, weights_path: Optional[str | Path]) -> None:
        """Load model weights when a custom checkpoint is available."""

        if not weights_path:
            return

        ckpt_path = Path(weights_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _build_with_cached_weights(weights_enum: ResNet50_Weights) -> models.ResNet:
        """Instantiate ResNet50 with weights fetched via certifi-backed HTTPS."""

        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        hub_dir.mkdir(parents=True, exist_ok=True)
        weight_path = hub_dir / Path(weights_enum.url).name

        if not weight_path.exists():
            if certifi is None:
                warnings.warn("certifi unavailable; using random init for ReID model")
                return models.resnet50(weights=None)
            try:
                _download_with_certifi(weights_enum.url, weight_path)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Failed to download ResNet50 weights ({exc}); using random init")
                return models.resnet50(weights=None)

        backbone = models.resnet50(weights=None)
        state_dict = torch.load(weight_path, map_location="cpu")
        backbone.load_state_dict(state_dict)
        return backbone


def build_model(
    embedding_dim: int = 512,
    pretrained: bool = True,
    weights_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
) -> ReIDModel:
    """Helper to initialize and optionally load checkpointed weights."""

    model = ReIDModel(embedding_dim=embedding_dim, pretrained=pretrained)
    model.load_checkpoint(weights_path)

    if device is not None:
        model = model.to(device)

    return model


def _download_with_certifi(url: str, destination: Path) -> None:
    """Download using certifi CA bundle to survive macOS Python SSL issues."""

    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, context=context) as response, destination.open("wb") as fp:
        shutil.copyfileobj(response, fp)
