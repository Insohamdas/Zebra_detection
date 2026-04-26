"""Flank classification for zebra identification (left vs right)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2

class FlankClassifier:
    """Classifies a zebra crop as 'left' or 'right' flank."""
    
    def __init__(self, model_path: str | None = None, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        # Use a lightweight MobileNetV3 for binary classification (left=0, right=1)
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def classify(self, image: np.ndarray) -> str:
        """Classify the flank side of a zebra crop.
        
        Args:
            image: BGR numpy array
            
        Returns:
            'left' or 'right'
        """
        # For now, if no weights are loaded, we use a simple heuristic or default
        # REAL IMPLEMENTATION:
        # tensor = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        # output = self.model(tensor)
        # prediction = torch.argmax(output, dim=1).item()
        # return "left" if prediction == 0 else "right"
        
        # HEURISTIC STUB: (placeholder until model is trained)
        # In many dataset conventions, zebras walking left -> left flank visible.
        # This is a placeholder; in production, use the trained model above.
        return "left"
