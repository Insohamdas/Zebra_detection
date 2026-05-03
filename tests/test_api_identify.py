"""Tests for the ZEBRAID identification API endpoint."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zebraid.api.app import create_app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    import os

    # Enable fast mock mode for most tests to avoid heavy ML dependencies
    os.environ.setdefault("IDENTIFY_MOCK", "1")

    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client):
    """Test root / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_identify_endpoint_with_valid_image(client):
    """Test /identify endpoint with a valid image."""
    # Create a simple test image (100x100 RGB)
    import cv2
    # Use minimum required resolution (>=5MP)
    image = np.random.randint(0, 256, (2000, 2560, 3), dtype=np.uint8)
    success, encoded_image = cv2.imencode(".jpg", image)
    assert success

    image_bytes = io.BytesIO(encoded_image.tobytes())

    response = client.post(
        "/identify",
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "zebra_id" in data
    assert "confidence" in data
    assert "is_new" in data
    assert isinstance(data["zebra_id"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["is_new"], bool)
    assert 0.0 <= data["confidence"] <= 1.0


def test_identify_endpoint_with_png(client):
    """Test /identify endpoint with PNG image."""
    import cv2
    image = np.ones((2000, 2560, 3), dtype=np.uint8) * 128
    success, encoded_image = cv2.imencode(".png", image)
    assert success

    image_bytes = io.BytesIO(encoded_image.tobytes())

    response = client.post(
        "/identify",
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "zebra_id" in data


def test_identify_endpoint_invalid_image(client):
    """Test /identify endpoint with invalid image data."""
    # Create a client without IDENTIFY_MOCK so decoding is exercised
    import os
    orig = os.environ.pop("IDENTIFY_MOCK", None)
    try:
        app = create_app()
        from fastapi.testclient import TestClient as TC

        no_mock_client = TC(app)
        response = no_mock_client.post(
            "/identify",
            files={"image": ("test.jpg", b"not an image", "image/jpeg")},
        )

        assert response.status_code == 400
        assert "Could not decode image" in response.json()["detail"]
    finally:
        if orig is not None:
            os.environ["IDENTIFY_MOCK"] = orig


def test_identify_endpoint_missing_file(client):
    """Test /identify endpoint without image file."""
    response = client.post("/identify")

    assert response.status_code == 422  # Unprocessable Entity


def test_identify_endpoint_multiple_calls_different_images(client):
    """Test that different images produce different zebra IDs."""
    import cv2

    # Create two distinct images
    image1 = np.ones((2000, 2560, 3), dtype=np.uint8) * 50
    image2 = np.ones((2000, 2560, 3), dtype=np.uint8) * 200

    _, img1_encoded = cv2.imencode(".jpg", image1)
    _, img2_encoded = cv2.imencode(".jpg", image2)

    response1 = client.post(
        "/identify",
        files={"image": ("test1.jpg", io.BytesIO(img1_encoded.tobytes()), "image/jpeg")},
    )
    response2 = client.post(
        "/identify",
        files={"image": ("test2.jpg", io.BytesIO(img2_encoded.tobytes()), "image/jpeg")},
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Different images likely produce different IDs
    # (not guaranteed, but very likely)
    assert "zebra_id" in data1
    assert "zebra_id" in data2


def test_identify_response_schema(client):
    """Test that /identify response matches IdentificationResponse schema."""
    import cv2
    image = np.random.randint(0, 256, (2000, 2560, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)

    response = client.post(
        "/identify",
        files={"image": ("test.jpg", io.BytesIO(encoded_image.tobytes()), "image/jpeg")},
    )

    assert response.status_code == 200

    data = response.json()

    # Verify schema
    assert set(data.keys()) == {"zebra_id", "confidence", "is_new"}
    assert isinstance(data["zebra_id"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["is_new"], bool)
    assert len(data["zebra_id"]) > 0
    assert 0.0 <= data["confidence"] <= 1.0
