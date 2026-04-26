import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class FeatureEncoder:
    """ResNet-50 feature encoder for zebra image embeddings."""

    def __init__(self, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        try:
            self.model = resnet50(pretrained=True)
        except Exception:
            self.model = resnet50(weights=None)

        # Specialized Re-ID head for identity-aware 128D embeddings
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract normalized embeddings from image tensor.
        
        Returns L2-normalized 128-dim identity embedding.
        """
        embedding = self.model(image_tensor.to(self.device))
        
        # Normalize to unit L2 norm for consistent distance metrics
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


def gabor_features(image: np.ndarray) -> np.ndarray:
    """Extract 32D Gabor texture features from image."""
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    gray = cv2.resize(gray, (128, 128))
    features = []
    
    # 4 orientations, 4 scales -> 16 filters. Mean and Variance -> 32 features
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 3, 5, 7]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(filtered.mean())
            features.append(filtered.var())
            
    # Normalize features to 0-1 roughly
    feats = np.array(features, dtype=np.float32)
    max_val = np.abs(feats).max()
    if max_val > 0:
        feats = feats / max_val
    return feats


def combine_features(global_vec: torch.Tensor, patches_vec: list[torch.Tensor], alpha: float = 0.7) -> torch.Tensor:
    """Concatenate global + patch features using balanced alpha weighting.
    
    Args:
        global_vec: Global features (ResNet embedding, e.g. 128D)
        patches_vec: List of patch feature tensors (e.g. Gabor features, 32D)
        alpha: Weight for global features vs patch features (default: 0.7)
    
    Returns:
        L2-normalized concatenated feature vector
    """
    if not patches_vec:
        return F.normalize(global_vec, p=2, dim=1)
    
    # Normalize global_vec
    g_norm = F.normalize(global_vec, p=2, dim=1)
    
    # Concatenate and normalize patch features
    patches = torch.cat(patches_vec, dim=1)
    p_norm = F.normalize(patches, p=2, dim=1)
    
    # Apply alpha weighting
    weighted_global = g_norm * alpha
    weighted_patches = p_norm * (1.0 - alpha)
    
    # Combine
    combined = torch.cat([weighted_global, weighted_patches], dim=1)
    
    # Final L2 normalization for matching engine consistency
    combined = F.normalize(combined, p=2, dim=1)
    
    return combined
