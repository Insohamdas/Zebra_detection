import cv2
import importlib
import numpy as np
import torch
from fastapi.testclient import TestClient

app_module = importlib.import_module("zebraid.api.app")
from zebraid.api.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ZEBRAID API foundation is ready"}


def test_identify_endpoint_uses_segmentation(monkeypatch) -> None:
    image = np.random.randint(0, 256, (2000, 2560, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success is True

    class DummySegmenter:
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, frame: np.ndarray, **kwargs) -> np.ndarray:
            self.calls += 1
            assert frame.shape == image.shape
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[:, : frame.shape[1] // 2] = 1
            return mask

    class DummyEncoder:
        def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
            assert image_tensor.shape == (1, 3, 256, 512)
            return torch.ones((1, 512), dtype=torch.float32)

    class DummyEngine:
        def match_with_confidence(self, embedding: np.ndarray, flank: str = "left", **kwargs) -> tuple[str, float, bool]:
            assert embedding.shape == (626,)
            return "ZEBRA-TEST", 0.95, False
            
    class DummyFlankClassifier:
        def classify(self, frame: np.ndarray) -> str:
            return "left"

    class DummyDetector:
        def detect_boxes(self, frame: np.ndarray) -> list[np.ndarray]:
            return [np.array([0, 0, frame.shape[1], frame.shape[0]], dtype=np.float32)]

    dummy_segmenter = DummySegmenter()
    dummy_encoder = DummyEncoder()
    dummy_engine = DummyEngine()
    dummy_flank = DummyFlankClassifier()

    monkeypatch.setattr(
        app_module,
        "get_pipeline",
        lambda: (object(), dummy_engine, dummy_encoder, dummy_segmenter, dummy_flank, DummyDetector()),
    )

    response = client.post(
        "/identify",
        files={"image": ("test.jpg", encoded.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "zebra_id": "ZEBRA-TEST",
        "confidence": 0.95,
        "is_new": False,
    }
    assert dummy_segmenter.calls == 1
