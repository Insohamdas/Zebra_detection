"""YOLO-based object detector to filter out non-zebra images."""

import logging
import cv2
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
        
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> bool:
        """Check if the image contains at least one zebra.
        
        Args:
            image: OpenCV BGR image
            conf_threshold: Minimum confidence to accept a detection
            
        Returns:
            True if a zebra is detected, False otherwise
        """
        try:
            # Run YOLO inference
            results = self.model(image, verbose=False)
            
            # Extract detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    # Get class name
                    class_name = self.model.names[cls_id].lower()
                    
                    # COCO class for zebra is usually 22, but we check name to be safe
                    if class_name == "zebra" and conf >= conf_threshold:
                        LOGGER.info(f"Zebra detected! Confidence: {conf:.2f}")
                        return True
                        
            LOGGER.warning("No zebra detected in the image.")
            return False
            
        except Exception as e:
            LOGGER.error(f"YOLO detection failed: {e}")
            # Fail open if detection crashes, or fail closed?
            # Safer to fail open so we don't block legitimate requests if YOLO acts up
            return True
