def test_debug_shapes():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from fastapi.testclient import TestClient
    from zebraid.api.app import app
    import numpy as np
    import cv2
    import io

    client = TestClient(app)
    image = np.random.randint(0, 256, (2000, 2560, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    image_bytes = io.BytesIO(encoded_image.tobytes())

    print("Sending request...")
    response = client.post(
        "/identify",
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )
    print("Response Status:", response.status_code)
    if response.status_code != 200:
        print(response.json())
        assert False, "FAILED"
