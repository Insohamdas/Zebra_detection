import numpy as np
import pytest
import torch
import torch.nn as nn

from zebraid.feature_engine import encoder as encoder_module
from zebraid.feature_engine.encoder import (
    FeatureEncoder,
    StripeStabilityIndex,
    combine_features,
    engineered_stripe_features,
    gabor_features,
    stripe_zone_stats,
    zone_gabor_features,
)


class DummyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
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
    # Check that it outputs 512 dimensions eventually
    assert isinstance(feature_encoder.model.fc[-1], nn.Linear)
    assert feature_encoder.model.fc[-1].out_features == 512


def test_feature_encoder_encode(feature_encoder):
    dummy_image = torch.randn(1, 3, 224, 224)

    feature_vector = feature_encoder.encode(dummy_image)

    assert isinstance(feature_vector, torch.Tensor)
    assert feature_vector.shape == (1, 512)
    
    # Check that the embedding is L2-normalized (unit norm)
    norm = torch.linalg.norm(feature_vector, ord=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-6)


def test_feature_encoder_encode_multiscale(feature_encoder):
    dummy_image = torch.randn(1, 3, 256, 512)

    feature_vector = feature_encoder.encode_multiscale(dummy_image)

    assert feature_vector.shape == (1, 1024)
    norm = torch.linalg.norm(feature_vector, ord=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-5)


def test_gabor_features():
    dummy_image = np.random.rand(100, 100).astype(np.float32)

    features = gabor_features(dummy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (32,)


def test_zone_gabor_features():
    dummy_image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)

    features = zone_gabor_features(dummy_image)

    assert features.shape == (96,)
    assert np.isclose(np.linalg.norm(features[:32]), 1.0, atol=1e-5)


def test_stripe_zone_stats():
    dummy_image = np.zeros((256, 512, 3), dtype=np.uint8)
    dummy_image[:, ::16] = 255

    stats = stripe_zone_stats(dummy_image)

    assert stats.shape == (18,)
    assert stats[0] > 0


def test_engineered_stripe_features():
    dummy_image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)

    features = engineered_stripe_features(dummy_image)

    assert features.shape == (114,)
    assert np.isclose(np.linalg.norm(features), 1.0, atol=1e-5)


def test_stripe_stability_index_masks_unstable_dimensions():
    features = np.array(
        [
            [[1.0, 0.1], [2.0, 9.0], [3.0, 0.2]],
            [[1.1, 3.5], [2.1, 0.4], [3.1, 4.0]],
            [[6.0, 0.2], [8.0, 9.5], [9.0, 0.3]],
            [[6.1, 3.7], [8.1, 0.6], [9.1, 4.1]],
        ],
        dtype=np.float32,
    )
    labels = np.array(["a", "a", "b", "b"])

    ssi = StripeStabilityIndex(threshold=0.4).fit(features, labels)
    masked = ssi.transform(features)

    assert ssi.ssi_.shape == (3, 2)
    assert ssi.mask_.shape == (3, 2)
    assert ssi.mask_[0, 0]
    assert not ssi.mask_[0, 1]
    assert np.all(masked[:, 0, 1] == 0.0)


def test_combine_features():
    # Use 2D tensors with batch size 1
    global_vec = torch.randn(1, 512)
    patches_vec = [torch.randn(1, 32) for _ in range(5)]

    combined = combine_features(global_vec, patches_vec, alpha=0.7)

    assert isinstance(combined, torch.Tensor)
    # Global (512) + 5 * Patches (32) = 512 + 160 = 672
    assert combined.shape == (1, 672)
    
    # Check concatenation and normalization
    norm = torch.linalg.norm(combined, ord=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-5)


def test_combine_features_no_patches():
    global_vec = torch.randn(1, 512)

    combined = combine_features(global_vec, [])

    # combined returns F.normalize(global_vec, p=2, dim=1)
    expected = torch.nn.functional.normalize(global_vec, p=2, dim=1)
    assert torch.allclose(combined, expected, atol=1e-5)
    assert combined.shape == (1, 512)


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
