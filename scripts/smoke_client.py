import numpy as np
from PIL import Image
import httpx

# Create a 5+ MP random color image (2000 x 2560 = 5,120,000 px)
h, w = 2000, 2560
img = (np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
Image.fromarray(img).save('smoke.jpg', 'JPEG', quality=90)

files = {'image': ('smoke.jpg', open('smoke.jpg', 'rb'), 'image/jpeg')}

with httpx.Client(timeout=60.0) as client:
    try:
        r = client.post('http://127.0.0.1:8000/identify', files=files)
        print('STATUS', r.status_code)
        try:
            print('JSON:', r.json())
        except Exception:
            print('TEXT:', r.text)
    except Exception as e:
        print('ERROR:', e)
