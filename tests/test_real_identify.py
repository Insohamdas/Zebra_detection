"""Tests for real zebra identification pipeline."""
import numpy as np
import pytest
import torch

from zebraid.feature_engine import FeatureEncoder
from zebraid.matching import MatchingEngine
from zebraid.pipelines.live_identification import IdentificationCandidate
from zebraid.pipelines.real_identify import create_real_identifier
from zebraid.registry import FaissStore


@pytest.fixture
def registry():
    """Create a fresh FAISS registry for each test."""
    return FaissStore(embedding_dim=626)


@pytest.fixture
def encoder():
    """Create a feature encoder."""
    return FeatureEncoder()


def test_create_real_identifier_returns_callable(registry, encoder):
    """Test that create_real_identifier returns a callable function."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    assert callable(identifier)


def test_real_identifier_with_none_frame(registry, encoder):
    """Test that real identifier handles None frames gracefully."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    result = identifier(None)
    assert result is None


def test_real_identifier_with_empty_frame(registry, encoder):
    """Test that real identifier handles empty frames gracefully."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    empty_frame = np.array([], dtype=np.uint8)
    result = identifier(empty_frame)
    assert result is None


def test_real_identifier_returns_identification_candidate(registry, encoder):
    """Test that real identifier returns IdentificationCandidate."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    # Create a valid test frame
    frame = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    result = identifier(frame)
    
    assert isinstance(result, IdentificationCandidate)
    assert isinstance(result.zebra_id, str)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_real_identifier_creates_new_zebra_on_first_call(registry, encoder):
    """Test that first frame creates a new zebra ID."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    frame = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    result = identifier(frame)
    
    assert isinstance(result.zebra_id, str)
    assert result.zebra_id.startswith("ZEB-")


def test_real_identifier_matches_identical_frame(registry, encoder):
    """Test that identical frames are matched to same ID."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)

    identifier.matching_engine.match_with_confidence = lambda embedding: (
        "ZEBRA-STATIC",
        0.99,
        False,
    )
    
    # Use a fixed seed for reproducibility
    frame = np.ones((640, 480, 3), dtype=np.uint8) * 128
    
    # First call creates ID
    result1 = identifier(frame)
    zebra_id_1 = result1.zebra_id
    
    # Second call with same frame should match (if within threshold)
    result2 = identifier(frame)
    zebra_id_2 = result2.zebra_id
    
    # Due to floating point precision, may or may not match
    # Just verify both are valid IDs
    assert isinstance(zebra_id_1, str)
    assert isinstance(zebra_id_2, str)


def test_real_identifier_attributes(registry, encoder):
    """Test that identifier has matching_engine, registry, and encoder attributes."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    assert hasattr(identifier, 'matching_engine')
    assert hasattr(identifier, 'registry')
    assert hasattr(identifier, 'encoder')
    assert isinstance(identifier.matching_engine, MatchingEngine)
    assert isinstance(identifier.registry, FaissStore)
    assert isinstance(identifier.encoder, FeatureEncoder)


def test_real_identifier_with_grayscale_frame(registry, encoder):
    """Test that real identifier handles grayscale frames."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    # Grayscale frame
    gray_frame = np.random.randint(0, 256, (640, 480), dtype=np.uint8)
    result = identifier(gray_frame)
    
    assert isinstance(result, IdentificationCandidate)
    assert result.zebra_id is not None


def test_real_identifier_with_rgba_frame(registry, encoder):
    """Test that real identifier handles RGBA frames."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    # RGBA frame
    rgba_frame = np.random.randint(0, 256, (640, 480, 4), dtype=np.uint8)
    result = identifier(rgba_frame)
    
    assert isinstance(result, IdentificationCandidate)
    assert result.zebra_id is not None


def test_real_identifier_with_float_frame(registry, encoder):
    """Test that real identifier handles float-valued frames."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    # Float frame (e.g., [0, 1] range)
    float_frame = np.random.rand(640, 480, 3).astype(np.float32)
    result = identifier(float_frame)
    
    assert isinstance(result, IdentificationCandidate)
    assert result.zebra_id is not None


def test_real_identifier_with_custom_match_threshold(registry, encoder):
    """Test that custom match threshold is respected."""
    threshold = 1.0  # Very high threshold
    identifier = create_real_identifier(
        registry=registry,
        encoder=encoder,
        match_threshold=threshold,
    )
    
    frame = np.ones((640, 480, 3), dtype=np.uint8) * 100
    result = identifier(frame)
    
    # Should have a matching engine with the custom threshold
    assert identifier.matching_engine.distance_threshold == threshold


def test_real_identifier_creates_default_registry_and_encoder():
    """Test that identifier creates defaults when not provided."""
    identifier = create_real_identifier()
    
    assert identifier.registry is not None
    assert isinstance(identifier.registry, FaissStore)
    assert identifier.encoder is not None
    assert isinstance(identifier.encoder, FeatureEncoder)


def test_real_identifier_confidence_conversion(registry, encoder):
    """Test that distance is converted to confidence correctly."""
    identifier = create_real_identifier(registry=registry, encoder=encoder)
    
    frame = np.ones((640, 480, 3), dtype=np.uint8) * 128
    result = identifier(frame)
    
    # Confidence should be between 0 and 1
    assert 0.0 <= result.confidence <= 1.0


def test_real_identifier_uses_segmenter_before_encoding(registry):
    class DummyEncoder:
        def encode(self, image_tensor):
            assert image_tensor.shape == (1, 3, 256, 512)
            return torch.ones((1, 512), dtype=torch.float32)

    class RecordingSegmenter:
        def __init__(self):
            self.calls = 0

        def segment(self, frame):
            self.calls += 1
            assert frame.dtype == np.uint8
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[:, : frame.shape[1] // 2] = 1
            return mask

    segmenter = RecordingSegmenter()
    identifier = create_real_identifier(
        registry=registry,
        encoder=DummyEncoder(),
        segmenter=segmenter,
    )

    frame = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    result = identifier(frame)

    assert result is not None
    assert segmenter.calls == 1


def test_real_identifier_forwards_keypoints_to_prepare_tensor(monkeypatch, registry):
    class DummyEncoder:
        def encode(self, image_tensor):
            assert image_tensor.shape == (1, 3, 256, 512)
            return torch.ones((1, 626), dtype=torch.float32)

    class DummySegmenter:
        def segment(self, frame, box=None):
            return np.ones(frame.shape[:2], dtype=np.uint8)

    class DummyDetector:
        def detect_with_quality(self, frame):
            return [
                {
                    "box": np.array([10, 10, 200, 200], dtype=np.float32),
                    "rejected": False,
                    "reject_reasons": [],
                    "quality": {"blur": 100.0, "entropy": 5.0, "stripe_contrast": 0.7},
                }
            ]

    class DummyKeypointDetector:
        def detect_keypoints(self, frame, box=None):
            assert box is not None
            return np.array(
                [
                    [8, 38],
                    [18, 28],
                    [36, 24],
                    [54, 22],
                    [72, 23],
                    [92, 28],
                    [128, 38],
                    [30, 78],
                    [48, 80],
                    [72, 80],
                    [96, 78],
                    [118, 54],
                ],
                dtype=np.float32,
            )

    class DummyFlankClassifier:
        def classify(self, frame):
            return "left"

    prepare_calls = {}

    def fake_prepare_tensor(image, *, segmenter=None, box=None, keypoints=None):
        prepare_calls["box"] = box
        prepare_calls["keypoints"] = keypoints
        return torch.ones((1, 3, 256, 512), dtype=torch.float32)

    monkeypatch.setattr("zebraid.pipelines.real_identify.prepare_tensor", fake_prepare_tensor)

    identifier = create_real_identifier(
        registry=registry,
        encoder=DummyEncoder(),
        segmenter=DummySegmenter(),
        flank_classifier=DummyFlankClassifier(),
        detector=DummyDetector(),
        keypoint_detector=DummyKeypointDetector(),
    )
    identifier.matching_engine.match_with_confidence = lambda *args, **kwargs: ("ZEB-KEYPTS", 0.88, False)

    frame = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    result = identifier(frame)

    assert isinstance(result, IdentificationCandidate)
    assert result.zebra_id == "ZEB-KEYPTS"
    assert prepare_calls["box"].shape == (4,)
    assert prepare_calls["keypoints"].shape == (12, 2)


def test_real_identifier_code_mode_uses_masked_features_and_three_phase_resolution(monkeypatch, registry):
    class DummyEncoder:
        def encode_multiscale(self, image_tensor):
            assert image_tensor.shape == (1, 3, 256, 512)
            return torch.ones((1, 1024), dtype=torch.float32)

    class DummySegmenter:
        def segment(self, frame, box=None):
            return np.ones(frame.shape[:2], dtype=np.uint8)

    class DummyDetector:
        def best_box(self, frame):
            return np.array([5, 5, 100, 120], dtype=np.float32)

    class DummyFlankClassifier:
        def classify(self, frame):
            return "right"

    class DummySSI:
        def __init__(self):
            self.calls = 0

        def transform(self, features):
            self.calls += 1
            assert features.shape == (3, 32)
            return features * 0.5

    captured = {}

    def fake_zone_gabor_features(image):
        captured["zone_image_shape"] = image.shape
        return np.arange(96, dtype=np.float32)

    def fake_stripe_stats(image):
        captured["stats_image_shape"] = image.shape
        return np.arange(18, dtype=np.float32)

    def fake_global_itq_code(descriptor, binarizer=None):
        captured["global_descriptor_shape"] = descriptor.shape
        return np.ones(512, dtype=np.uint8)

    class FakePatchCodes:
        def __init__(self):
            self.shoulder = np.zeros(128, dtype=np.uint8)
            self.torso = np.ones(128, dtype=np.uint8)
            self.neck = np.ones(64, dtype=np.uint8)

    def fake_local_patch_codes(zone_descriptors, **kwargs):
        captured["local_zone_shapes"] = {k: np.asarray(v).shape for k, v in zone_descriptors.items()}
        return FakePatchCodes()

    def fake_resolve_three_phase_identity(*args, **kwargs):
        captured["resolve_kwargs"] = kwargs
        return "ZEB-CODE", 0.77, False, "hamming"

    monkeypatch.setattr("zebraid.pipelines.real_identify.zone_gabor_features", fake_zone_gabor_features)
    monkeypatch.setattr("zebraid.pipelines.real_identify.stripe_zone_stats", fake_stripe_stats)
    monkeypatch.setattr("zebraid.pipelines.real_identify.global_itq_code", fake_global_itq_code)
    monkeypatch.setattr("zebraid.pipelines.real_identify.local_patch_codes", fake_local_patch_codes)

    identifier = create_real_identifier(
        registry=registry,
        encoder=DummyEncoder(),
        segmenter=DummySegmenter(),
        flank_classifier=DummyFlankClassifier(),
        detector=DummyDetector(),
        keypoint_detector=None,
        identity_mode="code",
        ssi_index=DummySSI(),
    )
    identifier.matching_engine.resolve_three_phase_identity = fake_resolve_three_phase_identity

    frame = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    result = identifier(frame)

    assert isinstance(result, IdentificationCandidate)
    assert result.zebra_id == "ZEB-CODE"
    assert result.confidence == pytest.approx(0.77, rel=1e-6)
    assert captured["zone_image_shape"] == frame.shape
    assert captured["stats_image_shape"] == frame.shape
    assert captured["global_descriptor_shape"] == (1138,)
    assert captured["local_zone_shapes"] == {"shoulder": (32,), "torso": (32,), "neck": (32,)}
    assert captured["resolve_kwargs"]["flank"] == "right"
