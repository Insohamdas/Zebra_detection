import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

BODY_ZONES = ("shoulder", "torso", "neck")
STRIPE_STATS_PER_ZONE = 6
GABOR_ORIENTATIONS = tuple(np.linspace(0, np.pi, 8, endpoint=False))
GABOR_FREQUENCIES = (0.08, 0.12, 0.18, 0.26)
MULTISCALE_PATCH_GRIDS = ((16, 32), (64, 128))


class FeatureEncoder:
    """ResNet-50 feature encoder for zebra image embeddings."""

    def __init__(
        self,
        device: str | None = None,
        checkpoint_path: str | None = None,
        embedding_dim: int = 512,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.model = resnet50(weights=None)

        # Re-ID head shape matches the paper target. It is only identity-
        # discriminative once a triplet-loss checkpoint is loaded.
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract normalized embeddings from image tensor.
        
        Returns L2-normalized identity embedding.
        """
        embedding = self.model(image_tensor.to(self.device))
        
        # Normalize to unit L2 norm for consistent distance metrics
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

    @torch.no_grad()
    def encode_multiscale(
        self,
        image_tensor: torch.Tensor,
        scales: tuple[tuple[int, int], ...] = MULTISCALE_PATCH_GRIDS,
    ) -> torch.Tensor:
        """Extract coarse + fine descriptors and concatenate to 1024D.

        ``scales`` are ``(height, width)`` tensors corresponding to the paper's
        32x16 coarse and 128x64 fine grid sizes.
        """

        image_tensor = image_tensor.to(self.device)
        descriptors = []
        for height, width in scales:
            scaled = F.interpolate(
                image_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            descriptors.append(self.encode(scaled))

        return F.normalize(torch.cat(descriptors, dim=1), p=2, dim=1)


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


def body_zones(image: np.ndarray) -> dict[str, np.ndarray]:
    """Split a side-view body image into shoulder, torso, and neck zones."""

    image_bgr = _coerce_uint8_image(image)
    width = image_bgr.shape[1]
    one_third = max(1, width // 3)
    return {
        "shoulder": image_bgr[:, :one_third],
        "torso": image_bgr[:, one_third : 2 * one_third],
        "neck": image_bgr[:, 2 * one_third :],
    }


def zone_gabor_features(image: np.ndarray) -> np.ndarray:
    """Extract 32 L2-normalized Gabor responses per body zone.

    The filter bank uses 8 orientations x 4 frequencies. Each filter
    contributes the mean absolute response in that zone.
    """

    features: list[np.ndarray] = []
    for zone in BODY_ZONES:
        zone_gray = _to_gray(body_zones(image)[zone])
        zone_gray = cv2.resize(zone_gray, (128, 128), interpolation=cv2.INTER_AREA)
        zone_features = []

        for theta in GABOR_ORIENTATIONS:
            for frequency in GABOR_FREQUENCIES:
                wavelength = 1.0 / frequency
                kernel = cv2.getGaborKernel(
                    (21, 21),
                    sigma=4.0,
                    theta=float(theta),
                    lambd=float(wavelength),
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F,
                )
                filtered = cv2.filter2D(zone_gray, cv2.CV_32F, kernel)
                zone_features.append(float(np.mean(np.abs(filtered))))

        zone_vec = _l2_normalize(np.asarray(zone_features, dtype=np.float32))
        features.append(zone_vec)

    return np.concatenate(features).astype(np.float32)


def stripe_zone_stats(
    image: np.ndarray,
    *,
    sigma: float = 1.4,
    low_threshold: float = 0.05,
    high_threshold: float = 0.15,
) -> np.ndarray:
    """Compute Canny connected-component stripe statistics per body zone.

    Returns ``(n, mean_width, width_variance, spacing, orientation, curvature)``
    for shoulder, torso, and neck, concatenated into an 18D vector.
    """

    stats: list[float] = []
    for zone_img in body_zones(image).values():
        gray = _to_gray(zone_img)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        edges = cv2.Canny(
            blurred,
            threshold1=int(low_threshold * 255),
            threshold2=int(high_threshold * 255),
            L2gradient=True,
        )

        components, centers, widths, orientations, curvatures = _stripe_components(edges)
        n_stripes = float(len(components))
        mean_width = float(np.mean(widths)) if widths else 0.0
        width_var = float(np.var(widths)) if widths else 0.0
        spacing = float(np.mean(np.diff(sorted(centers)))) if len(centers) > 1 else 0.0
        orientation = float(np.mean(orientations)) if orientations else 0.0
        curvature = float(np.mean(curvatures)) if curvatures else 0.0
        stats.extend([n_stripes, mean_width, width_var, spacing, orientation, curvature])

    return np.asarray(stats, dtype=np.float32)


def engineered_stripe_features(image: np.ndarray) -> np.ndarray:
    """Return normalized per-zone Gabor + Canny stripe features."""

    gabor = zone_gabor_features(image)
    stats = stripe_zone_stats(image)
    return _l2_normalize(np.concatenate([gabor, stats]).astype(np.float32))


class StripeStabilityIndex:
    """Mask unstable stripe features using SSI(z,d).

    SSI is computed as ``1 - sigma_within(z,d) / sigma_between(z,d)`` for each
    zone and feature dimension. Dimensions below ``threshold`` are masked.
    """

    def __init__(self, threshold: float = 0.4, eps: float = 1e-6) -> None:
        self.threshold = threshold
        self.eps = eps
        self.ssi_: np.ndarray | None = None
        self.mask_: np.ndarray | None = None

    def fit(self, zone_features: np.ndarray, labels: np.ndarray | list[str]) -> "StripeStabilityIndex":
        """Fit SSI from ``(sample, zone, feature)`` arrays and identity labels."""

        features = np.asarray(zone_features, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError("zone_features must have shape (samples, zones, features)")
        if features.shape[0] < 2:
            raise ValueError("at least two samples are required to fit SSI")

        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != features.shape[0]:
            raise ValueError("labels length must match number of samples")

        unique_labels = np.unique(labels_arr)
        if unique_labels.shape[0] < 2:
            raise ValueError("at least two identities are required to fit SSI")

        within_terms = []
        means = []
        for label in unique_labels:
            group = features[labels_arr == label]
            if group.shape[0] == 0:
                continue
            means.append(group.mean(axis=0))
            if group.shape[0] > 1:
                within_terms.append(group.std(axis=0))
            else:
                within_terms.append(np.zeros(features.shape[1:], dtype=np.float32))

        sigma_within = np.mean(np.stack(within_terms, axis=0), axis=0)
        sigma_between = np.std(np.stack(means, axis=0), axis=0)

        self.ssi_ = 1.0 - (sigma_within / (sigma_between + self.eps))
        self.mask_ = self.ssi_ >= self.threshold
        return self

    def transform(self, zone_features: np.ndarray) -> np.ndarray:
        """Apply the learned SSI mask to zone features."""

        if self.mask_ is None:
            raise RuntimeError("StripeStabilityIndex must be fit before transform")

        features = np.asarray(zone_features, dtype=np.float32)
        if features.shape[-2:] != self.mask_.shape:
            raise ValueError("zone feature dimensions do not match fitted SSI mask")

        return features * self.mask_.astype(np.float32)

    def fit_transform(self, zone_features: np.ndarray, labels: np.ndarray | list[str]) -> np.ndarray:
        """Fit SSI and return masked zone features."""

        return self.fit(zone_features, labels).transform(zone_features)


def _stripe_components(edges: np.ndarray) -> tuple[list[np.ndarray], list[float], list[float], list[float], list[float]]:
    """Extract connected edge components and derived stripe measurements."""

    num_labels, labels, component_stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    components: list[np.ndarray] = []
    centers: list[float] = []
    widths: list[float] = []
    orientations: list[float] = []
    curvatures: list[float] = []

    for label in range(1, num_labels):
        area = int(component_stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue

        points_yx = np.column_stack(np.where(labels == label))
        if points_yx.shape[0] < 2:
            continue

        points = points_yx[:, ::-1].astype(np.float32)
        components.append(points)
        centers.append(float(centroids[label][0]))

        x, y, w, h, _ = component_stats[label]
        widths.append(float(min(max(w, 1), max(h, 1))))

        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        orientations.append(float(np.arctan2(direction[1], direction[0])))

        ordered = points[np.argsort(points[:, 0])]
        if ordered.shape[0] >= 3:
            deltas = np.diff(ordered, axis=0)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            curvatures.append(float(np.mean(np.abs(np.diff(angles)))))
        else:
            curvatures.append(0.0)

    return components, centers, widths, orientations, curvatures


def _to_gray(image: np.ndarray) -> np.ndarray:
    image_u8 = _coerce_uint8_image(image)
    if image_u8.ndim == 2:
        return image_u8
    return cv2.cvtColor(image_u8, cv2.COLOR_BGR2GRAY)


def _coerce_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError("image must be grayscale, BGR, or BGRA")


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(features))
    if norm < 1e-8:
        return features.astype(np.float32)
    return (features / norm).astype(np.float32)


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
