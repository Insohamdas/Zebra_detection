import numpy as np
import torch

from zebraid.preprocessing.pipeline import (
	ZebraSegmenter,
	apply_mask,
	enhance,
	extract_patches,
	load_sam_model,
	normalize_pose,
	process_image,
	prepare_tensor,
	segment_and_clean,
)


def _sample_bgr_image() -> np.ndarray:
	image = np.zeros((96, 144, 3), dtype=np.uint8)
	image[:, :48] = [50, 80, 120]
	image[:, 48:96] = [90, 140, 190]
	image[:, 96:] = [30, 50, 90]
	return image


def test_apply_mask_preserves_shape() -> None:
	image = _sample_bgr_image()
	mask = np.zeros((96, 144), dtype=np.uint8)
	mask[:, :72] = 1

	masked = apply_mask(image, mask)
	assert masked.shape == image.shape
	assert np.all(masked[:, 80:] == 0)


def test_enhance_and_normalize_pose_shapes() -> None:
	image = _sample_bgr_image()
	enhanced = enhance(image)
	normalized = normalize_pose(enhanced)

	assert enhanced.shape == image.shape
	assert normalized.shape == (256, 256, 3)


def test_extract_patches_returns_three_regions() -> None:
	image = np.zeros((256, 256, 3), dtype=np.uint8)
	patches = extract_patches(image)

	assert set(patches.keys()) == {"shoulder", "torso", "neck"}
	assert patches["shoulder"].shape[0] == 256
	assert patches["torso"].shape[0] == 256
	assert patches["neck"].shape[0] == 256


def test_process_image_full_pipeline() -> None:
	image = _sample_bgr_image()

	segmenter = ZebraSegmenter(model=lambda img: np.ones(img.shape[:2], dtype=np.uint8))
	normalized, patches = process_image(image, segmenter=segmenter)

	assert normalized.shape == (256, 256, 3)
	assert normalized.dtype == np.uint8
	assert set(patches.keys()) == {"shoulder", "torso", "neck"}


def test_segment_and_clean_and_prepare_tensor_use_mask() -> None:
	image = _sample_bgr_image()

	class HalfMaskSegmenter:
		def segment(self, image: np.ndarray) -> np.ndarray:
			mask = np.zeros(image.shape[:2], dtype=np.uint8)
			mask[:, :72] = 1
			return mask

	clean = segment_and_clean(image, segmenter=HalfMaskSegmenter())
	tensor = prepare_tensor(image, segmenter=HalfMaskSegmenter())

	assert clean.shape == (256, 256, 3)
	assert clean.dtype == np.uint8
	assert tensor.shape == (1, 3, 256, 256)
	assert tensor.dtype == torch.float32


def test_load_sam_model_falls_back_when_checkpoint_missing() -> None:
	image = _sample_bgr_image()
	segment = load_sam_model(backend="sam", checkpoint_path="/tmp/does-not-exist.pt")
	mask = segment(image)

	assert mask.shape == image.shape[:2]
	assert mask.dtype == np.uint8


def test_load_sam_model_raises_without_fallback() -> None:
	try:
		load_sam_model(
			backend="sam",
			checkpoint_path="/tmp/does-not-exist.pt",
			fallback_to_otsu=False,
		)
		assert False, "expected FileNotFoundError"
	except FileNotFoundError:
		assert True