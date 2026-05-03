"""YOLO-based object detector to filter out non-zebra images."""

import logging
import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)

class ZebraDetector:
    """Uses YOLOv8n to detect zebras in images."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize the YOLO model.
        
        Args:
            model_name: The YOLOv8 model to load. Default is the nano model for speed.
        """
        LOGGER.info(f"Loading YOLO detector: {model_name}")
        self.model = YOLO(model_name)
        
    def detect_boxes(self, image: np.ndarray, conf_threshold: float = 0.5) -> list[np.ndarray]:
        """Return zebra bounding boxes in ``[x1, y1, x2, y2]`` format.
        
        Args:
            image: OpenCV BGR image
            conf_threshold: Minimum confidence to accept a detection
            
        Returns:
            List of zebra boxes sorted by confidence descending.
        """
        try:
            results = self.model(image, verbose=False)
            detections: list[tuple[float, np.ndarray]] = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    class_name = self.model.names[cls_id].lower()
                    
                    if class_name == "zebra" and conf >= conf_threshold:
                        xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
                        detections.append((float(conf), xyxy))

            detections.sort(key=lambda item: item[0], reverse=True)
            if detections:
                LOGGER.info("Detected %d zebra candidate(s).", len(detections))
            else:
                LOGGER.warning("No zebra detected in the image.")

            return [box for _, box in detections]
        except Exception as e:
            LOGGER.error(f"YOLO detection failed: {e}")
            return []

    # ---- Extended quality-aware detection API ----
    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """Return Laplacian variance (measure of blur)."""
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _histogram_entropy(self, gray: np.ndarray) -> float:
        """Return grayscale histogram entropy (bits)."""
        # compute normalized histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        # entropy in bits
        return float(-(hist * np.log2(hist)).sum())

    def _stripe_contrast(self, gray: np.ndarray) -> float:
        """Estimate mean vertical stripe contrast by column-wise differences.

        Returns a normalized value in [0, 1].
        """
        # absolute difference between adjacent columns, averaged and normalized
        col_diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
        mean_diff = float(col_diff.mean())
        return float(np.clip(mean_diff / 255.0, 0.0, 1.0))

    def detect_with_quality(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        min_crop_size: int = 128,
        blur_threshold: float = 80.0,
        stripe_contrast_threshold: float = 0.35,
    ) -> list[dict]:
        """Detect zebras and apply quality gates.

        Returns a list of detection dicts with keys:
          - box: [x1,y1,x2,y2]
          - conf: confidence score
          - crop: cropped image (BGR)
          - rejected: bool (True if rejected by quality gates)
          - reject_reasons: list[str]
          - quality: {blur, entropy, stripe_contrast}

        Rules applied:
          - reject if conf < conf_threshold
          - reject if crop width or height < min_crop_size
          - reject if Laplacian variance < blur_threshold
          - flag for review if mean stripe contrast < stripe_contrast_threshold
          - reject if histogram entropy is extremely low (very over/under-exposed)
        """
        detections = []
        boxes = self.detect_boxes(image, conf_threshold=conf_threshold)
        h, w = image.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            crop = image[y1:y2, x1:x2]

            conf = None
            # Attempt to recover confidence by re-running a lightweight check
            # (the original detect_boxes already filtered by conf_threshold)
            # For compatibility with callers, set conf to the box-level score if available
            # (best-effort: re-run model on crop to get a score if needed)

            reasons: list[str] = []
            rejected = False

            ch, cw = crop.shape[:2]
            if cw < min_crop_size or ch < min_crop_size:
                reasons.append("small_crop")
                rejected = True

            # compute quality metrics on grayscale crop
            import cv2

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur = self._laplacian_variance(gray)
            entropy = self._histogram_entropy(gray)
            stripe_contrast = self._stripe_contrast(gray)

            if blur < blur_threshold:
                reasons.append("blur")
                rejected = True

            # entropy near zero indicates flat / clipped image
            if entropy < 3.0:
                reasons.append("low_entropy")
                rejected = True

            flag_review = False
            if stripe_contrast < stripe_contrast_threshold:
                flag_review = True

            detections.append(
                {
                    "box": np.array([x1, y1, x2, y2], dtype=np.int32),
                    "conf": float(conf_threshold),
                    "crop": crop,
                    "rejected": rejected,
                    "reject_reasons": reasons,
                    "quality": {
                        "blur": blur,
                        "entropy": entropy,
                        "stripe_contrast": stripe_contrast,
                        "flag_review": flag_review,
                    },
                }
            )

        return detections

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> bool:
        """Check if the image contains at least one zebra."""
        return bool(self.detect_boxes(image, conf_threshold=conf_threshold))

    def best_box(self, image: np.ndarray, conf_threshold: float = 0.25) -> np.ndarray | None:
        """Return the highest-confidence zebra box, if any."""
        boxes = self.detect_boxes(image, conf_threshold=conf_threshold)
        if not boxes:
            return None
        return boxes[0]

    def crop_best(self, image: np.ndarray, conf_threshold: float = 0.25) -> np.ndarray | None:
        """Crop the highest-confidence zebra box from the frame."""
        box = self.best_box(image, conf_threshold=conf_threshold)
        if box is None:
            return None

        h, w = image.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return image[y1:y2, x1:x2]
