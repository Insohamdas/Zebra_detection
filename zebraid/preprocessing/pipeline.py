"""Image preprocessing pipeline for zebra identification."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
SegmentationBackend = Literal["otsu", "sam", "sam2"]
CanonicalSize = tuple[int, int]

# ImageNet mean in BGR format for background filling
IMAGENET_MEAN_BGR = (103.53, 116.28, 123.675)
CANONICAL_TEMPLATE_SIZE: CanonicalSize = (512, 256)

# Twelve side-view anatomical template points in normalized (x, y) order.
# The points are a canonical target layout for keypoint-driven warping; they are
# only used when a caller provides detected keypoints.
CANONICAL_SIDE_VIEW_KEYPOINTS = np.array(
	[
		[0.08, 0.40],
		[0.16, 0.30],
		[0.28, 0.25],
		[0.42, 0.23],
		[0.58, 0.24],
		[0.74, 0.30],
		[0.90, 0.40],
		[0.22, 0.82],
		[0.36, 0.84],
		[0.54, 0.84],
		[0.72, 0.82],
		[0.84, 0.56],
	],
	dtype=np.float32,
)



def _coerce_bgr_uint8(image: np.ndarray) -> np.ndarray:
	"""Convert common image layouts to a 3-channel BGR uint8 image."""

	if image.ndim == 2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	elif image.ndim == 3 and image.shape[2] == 4:
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	elif image.ndim != 3 or image.shape[2] != 3:
		raise ValueError("image must be grayscale or BGR/RGBA")

	if image.dtype != np.uint8:
		if np.issubdtype(image.dtype, np.floating):
			if float(image.max(initial=0.0)) <= 1.0:
				image = image * 255.0
		image = np.clip(image, 0, 255).astype(np.uint8)

	return image


def _default_device() -> str:
	"""Select best-available inference device."""

	try:
		import torch

		if torch.cuda.is_available():
			return "cuda"
		has_mps = getattr(torch.backends, "mps", None)
		if has_mps is not None and torch.backends.mps.is_available():
			return "mps"
	except Exception:
		pass

	return "cpu"


def _fallback_segmenter() -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
	"""Simple Otsu-based fallback segmentation."""

	def fallback_segment(image: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
		if image.ndim == 2:
			gray = image
		elif image.ndim == 3 and image.shape[2] in (3, 4):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			raise ValueError("image must be grayscale or BGR/RGBA")

		_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return (binary > 0).astype(np.uint8)

	return fallback_segment


def _load_sam_v1_predictor(
	*,
	checkpoint_path: str,
	model_type: str,
	device: str,
) -> "SamPredictor":
	"""Load Meta Segment Anything (SAM v1) predictor."""

	from segment_anything import SamPredictor, sam_model_registry

	sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
	sam.to(device=device)
	return SamPredictor(sam)


def _load_sam_v1_segmenter(
	*,
	checkpoint_path: str,
	model_type: str,
	device: str,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
	"""Load Meta Segment Anything (SAM v1) automatic mask generator or predictor wrapper."""

	from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

	sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
	sam.to(device=device)
	generator = SamAutomaticMaskGenerator(sam)

	def sam_segment(image: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		# If a box is provided, we prefer SamPredictor (handled in ZebraSegmenter)
		# This fallback is for when the segmenter is called as a simple callable
		masks = generator.generate(image_rgb)
		if not masks:
			return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

		largest = max(masks, key=lambda m: float(m.get("area", 0.0)))
		seg = largest.get("segmentation")
		if seg is None:
			return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
		return np.asarray(seg, dtype=np.uint8)

	return sam_segment


def _load_sam2_segmenter(
	*,
	checkpoint_path: str,
	model_type: str,
	device: str,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
	"""Load SAM2 automatic mask generator when available."""

	from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
	from sam2.build_sam import build_sam2

	model = build_sam2(model_type=model_type, checkpoint=checkpoint_path, device=device)
	generator = SAM2AutomaticMaskGenerator(model)

	def sam2_segment(image: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		masks = generator.generate(image_rgb)
		if not masks:
			return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

		largest = max(masks, key=lambda m: float(m.get("area", 0.0)))
		seg = largest.get("segmentation")
		if seg is None:
			return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
		return np.asarray(seg, dtype=np.uint8)

	return sam2_segment


def load_sam_model(
	*,
	backend: SegmentationBackend = "otsu",
	checkpoint_path: str | None = None,
	model_type: str = "vit_b",
	device: str | None = None,
	fallback_to_otsu: bool = True,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
	"""Load a segmentation model callable.

	Args:
		backend: ``otsu`` (default), ``sam``, or ``sam2``.
		checkpoint_path: Optional checkpoint path for SAM/SAM2 backends.
		model_type: Model identifier used by the selected backend.
		device: Inference device. If omitted, auto-selects cuda/mps/cpu.
		fallback_to_otsu: If true, gracefully fallback when model loading fails.
	"""

	if backend == "otsu":
		return _fallback_segmenter()

	if device is None:
		device = _default_device()

	try:
		if checkpoint_path is None:
			raise FileNotFoundError(
				"checkpoint_path is required for SAM/SAM2 backends"
			)

		checkpoint = Path(checkpoint_path)
		if not checkpoint.exists() or not checkpoint.is_file():
			raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

		if backend == "sam":
			return _load_sam_v1_segmenter(
				checkpoint_path=str(checkpoint),
				model_type=model_type,
				device=device,
			)

		return _load_sam2_segmenter(
			checkpoint_path=str(checkpoint),
			model_type=model_type,
			device=device,
		)
	except Exception as exc:
		if not fallback_to_otsu:
			raise
		LOGGER.warning(
			"Failed to load %s segmenter (%s). Falling back to Otsu thresholding.",
			backend,
			exc,
		)
		return _fallback_segmenter()


class ZebraSegmenter:
	def __init__(
		self,
		model: Callable[[np.ndarray, np.ndarray | None], np.ndarray] | None = None,
		*,
		backend: SegmentationBackend = "otsu",
		checkpoint_path: str | None = None,
		model_type: str = "vit_b",
		device: str | None = None,
		fallback_to_otsu: bool = True,
	):
		self.backend = backend
		self.predictor = None
		self.model = model
		
		if self.model is None:
			if self.backend == "sam" and checkpoint_path:
				try:
					dev = device or _default_device()
					self.predictor = _load_sam_v1_predictor(
						checkpoint_path=checkpoint_path,
						model_type=model_type,
						device=dev
					)
				except Exception as exc:
					LOGGER.warning(f"Failed to load SAM predictor ({exc}). Falling back to generator.")
			
			self.model = load_sam_model(
				backend=backend,
				checkpoint_path=checkpoint_path,
				model_type=model_type,
				device=device,
				fallback_to_otsu=fallback_to_otsu,
			)

	def segment(self, image: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
		"""Segment image, optionally using a bounding box prompt for SAM."""
		if self.predictor is not None and box is not None:
			image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			self.predictor.set_image(image_rgb)
			masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
			return masks[0].astype(np.uint8)
		
		return self.model(image)


def segment_and_clean(
	image: np.ndarray,
	*,
	segmenter: ZebraSegmenter | None = None,
	box: np.ndarray | None = None,
	keypoints: np.ndarray | None = None,
) -> np.ndarray:
	"""Segment a zebra and return a cleaned BGR image ready for encoding."""

	segmenter = segmenter or ZebraSegmenter(backend="sam")
	image_bgr = _coerce_bgr_uint8(image)
	try:
		mask = segmenter.segment(image_bgr, box=box)
	except TypeError:
		if box is not None:
			raise
		mask = segmenter.segment(image_bgr)

	if mask is None or not np.any(mask):
		LOGGER.warning("Segmentation produced an empty mask; using the original image")
		cleaned = image_bgr
	else:
		# Use ImageNet mean for background to minimize background influence on embeddings
		cleaned = apply_mask(image_bgr, mask, fill_color=IMAGENET_MEAN_BGR)

	cleaned = enhance(cleaned)
	return normalize_pose(cleaned, keypoints=keypoints)


def prepare_tensor(
	image: np.ndarray,
	*,
	segmenter: ZebraSegmenter | None = None,
	box: np.ndarray | None = None,
	keypoints: np.ndarray | None = None,
) -> "torch.Tensor":
	"""Segment, clean, resize, and convert an image into an encoder tensor."""

	import torch

	clean = segment_and_clean(image, segmenter=segmenter, box=box, keypoints=keypoints)
	clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
	return torch.from_numpy(clean_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def apply_mask(image: np.ndarray, mask: np.ndarray, fill_color: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
	"""Apply segmentation mask to image, filling background with fill_color."""

	if image.ndim not in (2, 3):
		raise ValueError("image must be 2D or 3D")

	if mask.ndim == 3:
		mask = mask[:, :, 0]

	if image.shape[:2] != mask.shape[:2]:
		mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

	# Create a copy to avoid modifying the original image
	output = image.copy()
	
	# Apply mask: pixels outside mask are filled with fill_color
	binary_mask = (mask > 0)
	if output.ndim == 3:
		output[~binary_mask] = fill_color
	else:
		output[~binary_mask] = fill_color[0] if isinstance(fill_color, (tuple, list)) else fill_color

	return output


def enhance(image: np.ndarray) -> np.ndarray:
	"""Enhance image contrast using CLAHE in LAB color space."""

	if image.ndim != 3 or image.shape[2] != 3:
		raise ValueError("enhance expects a 3-channel BGR image")

	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	clahe = cv2.createCLAHE()
	lab[:, :, 0] = clahe.apply(lab[:, :, 0])
	return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_pose(
	image: np.ndarray,
	*,
	keypoints: np.ndarray | None = None,
	canonical_size: CanonicalSize = CANONICAL_TEMPLATE_SIZE,
) -> np.ndarray:
	"""Normalize pose to the canonical side-view template.

	When 12 anatomical keypoints are supplied, the image is warped toward the
	canonical template using an affine transform. Without keypoints, this falls
	back to a deterministic resize to the same canonical dimensions.
	"""

	width, height = canonical_size
	if keypoints is None:
		return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

	points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
	if points.shape[0] != CANONICAL_SIDE_VIEW_KEYPOINTS.shape[0]:
		raise ValueError("keypoints must contain 12 anatomical points")

	target = CANONICAL_SIDE_VIEW_KEYPOINTS.copy()
	target[:, 0] *= width
	target[:, 1] *= height

	transform, _ = cv2.estimateAffinePartial2D(points, target, method=cv2.LMEDS)
	if transform is None:
		LOGGER.warning("Pose normalization failed; falling back to canonical resize")
		return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

	return cv2.warpAffine(
		image,
		transform,
		(width, height),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=IMAGENET_MEAN_BGR,
	)


def extract_patches(image: np.ndarray) -> dict[str, np.ndarray]:
	"""Extract coarse body region patches: shoulder, torso, neck."""

	if image.ndim != 3:
		raise ValueError("extract_patches expects a 3-channel image")

	_, w, _ = image.shape
	if w < 3:
		raise ValueError("image width must be at least 3 to extract patches")

	one_third = w // 3
	return {
		"shoulder": image[:, :one_third],
		"torso": image[:, one_third : 2 * one_third],
		"neck": image[:, 2 * one_third :],
	}


def process_image(
	image: np.ndarray,
	*,
	segmenter: ZebraSegmenter | None = None,
	keypoints: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
	"""Run the full preprocessing pipeline and return normalized image + patches."""

	normalized = segment_and_clean(image, segmenter=segmenter, keypoints=keypoints)
	patches = extract_patches(normalized)
	return normalized, patches
