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
        
    def detect_boxes(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[np.ndarray]:
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
