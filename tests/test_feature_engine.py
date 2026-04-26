import numpy as np
import pytest
import torch
import torch.nn as nn

from zebraid.feature_engine import encoder as encoder_module
from zebraid.feature_engine.encoder import (
    FeatureEncoder,
    combine_features,
    gabor_features,
)


class DummyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        x = torch.full((batch_size, 2048), 1.0, dtype=image.dtype, device=image.device)
        return self.fc(x)


@pytest.fixture()
def mock_resnet50(monkeypatch):
    def fake_resnet50(*args, **kwargs):
        return DummyResNet()

    monkeypatch.setattr(encoder_module, "resnet50", fake_resnet50)
    return fake_resnet50


@pytest.fixture()
def feature_encoder(mock_resnet50):
    return FeatureEncoder(device="cpu")


def test_feature_encoder_initialization(feature_encoder):
    assert feature_encoder is not None
    assert hasattr(feature_encoder, "model")
    assert feature_encoder.device == torch.device("cpu")
    assert isinstance(feature_encoder.model.fc, nn.Sequential)
    # Check that it outputs 128 dimensions eventually
    assert isinstance(feature_encoder.model.fc[-1], nn.Linear)
    assert feature_encoder.model.fc[-1].out_features == 128


def test_feature_encoder_encode(feature_encoder):
    dummy_image = torch.randn(1, 3, 224, 224)

    feature_vector = feature_encoder.encode(dummy_image)

    assert isinstance(feature_vector, torch.Tensor)
    assert feature_vector.shape == (1, 128)
    
    # Check that the embedding is L2-normalized (unit norm)
    norm = torch.linalg.norm(feature_vector, ord=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-6)


def test_gabor_features():
    dummy_image = np.random.rand(100, 100).astype(np.float32)

    features = gabor_features(dummy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (32,)


def test_combine_features():
    # Use 2D tensors with batch size 1
    global_vec = torch.randn(1, 128)
    patches_vec = [torch.randn(1, 32) for _ in range(5)]

    combined = combine_features(global_vec, patches_vec, alpha=0.7)

    assert isinstance(combined, torch.Tensor)
    # Global (128) + 5 * Patches (32) = 128 + 160 = 288
    assert combined.shape == (1, 288)
    
    # Check concatenation and normalization
    norm = torch.linalg.norm(combined, ord=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-5)


def test_combine_features_no_patches():
    global_vec = torch.randn(1, 128)

    combined = combine_features(global_vec, [])

    # combined returns F.normalize(global_vec, p=2, dim=1)
    expected = torch.nn.functional.normalize(global_vec, p=2, dim=1)
    assert torch.allclose(combined, expected, atol=1e-5)
    assert combined.shape == (1, 128)


def test_feature_encoder_device_selection(mock_resnet50):
    if torch.cuda.is_available():
        encoder = FeatureEncoder(device="cuda")
        assert encoder.device == torch.device("cuda")
    elif torch.backends.mps.is_available():
        encoder = FeatureEncoder(device="mps")
        assert encoder.device == torch.device("mps")
    else:
        encoder = FeatureEncoder(device="cpu")
        assert encoder.device == torch.device("cpu")

    auto_encoder = FeatureEncoder()
    if torch.cuda.is_available():
        assert auto_encoder.device == torch.device("cuda")
    elif torch.backends.mps.is_available():
        assert auto_encoder.device == torch.device("mps")
    else:
        assert auto_encoder.device == torch.device("cpu")
