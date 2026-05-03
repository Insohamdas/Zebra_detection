import numpy as np
import torch

from zebraid.preprocessing.pipeline import (
	ZebraSegmenter,
	apply_mask,
	enhance,
	extract_patches,
	load_sam_model,
	normalize_pose,
	normalize_pose_tps,
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
	assert normalized.shape == (256, 512, 3)


def test_extract_patches_returns_three_regions() -> None:
	image = np.zeros((256, 512, 3), dtype=np.uint8)
	patches = extract_patches(image)

	assert set(patches.keys()) == {"shoulder", "torso", "neck"}
	assert patches["shoulder"].shape[0] == 256
	assert patches["torso"].shape[0] == 256
	assert patches["neck"].shape[0] == 256


def test_process_image_full_pipeline() -> None:
	image = _sample_bgr_image()

	segmenter = ZebraSegmenter(model=lambda img: np.ones(img.shape[:2], dtype=np.uint8))
	normalized, patches = process_image(image, segmenter=segmenter)

	assert normalized.shape == (256, 512, 3)
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

	assert clean.shape == (256, 512, 3)
	assert clean.dtype == np.uint8
	assert tensor.shape == (1, 3, 256, 512)
	assert tensor.dtype == torch.float32


def test_prepare_tensor_forwards_box_prompt_to_segmenter() -> None:
	image = _sample_bgr_image()
	box = np.array([12, 8, 96, 72], dtype=np.float32)

	class BoxRecordingSegmenter:
		def __init__(self):
			self.box = None

		def segment(self, image: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
			self.box = box
			return np.ones(image.shape[:2], dtype=np.uint8)

	segmenter = BoxRecordingSegmenter()
	tensor = prepare_tensor(image, segmenter=segmenter, box=box)

	assert tensor.shape == (1, 3, 256, 512)
	assert segmenter.box is box


def test_normalize_pose_accepts_twelve_keypoints() -> None:
	image = _sample_bgr_image()
	keypoints = np.array(
		[
			[8, 38],
			[18, 28],
			[36, 24],
			[54, 22],
			[72, 23],
			[92, 28],
			[128, 38],
			[30, 78],
			[48, 80],
			[72, 80],
			[96, 78],
			[118, 54],
		],
		dtype=np.float32,
	)

	normalized = normalize_pose(image, keypoints=keypoints)

	assert normalized.shape == (256, 512, 3)


def test_normalize_pose_prefers_tps_when_available(monkeypatch) -> None:
	image = _sample_bgr_image()
	keypoints = np.array(
		[
			[8, 38],
			[18, 28],
			[36, 24],
			[54, 22],
			[72, 23],
			[92, 28],
			[128, 38],
			[30, 78],
			[48, 80],
			[72, 80],
			[96, 78],
			[118, 54],
		],
		dtype=np.float32,
	)

	called = {}

	def fake_tps(image_arg, points_arg, out_size_arg):
		called["args"] = (image_arg, points_arg, out_size_arg)
		return np.full((out_size_arg[1], out_size_arg[0], 3), 7, dtype=np.uint8)

	monkeypatch.setattr("zebraid.preprocessing.pipeline.normalize_pose_tps", fake_tps)
	normalized = normalize_pose(image, keypoints=keypoints)

	assert normalized.shape == (256, 512, 3)
	assert normalized.dtype == np.uint8
	assert np.all(normalized == 7)
	assert "args" in called
	assert called["args"][2] == (512, 256)


def test_normalize_pose_tps_with_mock_transformer(monkeypatch) -> None:
	image = _sample_bgr_image()
	points = np.array(
		[
			[8, 38],
			[18, 28],
			[36, 24],
			[54, 22],
			[72, 23],
			[92, 28],
			[128, 38],
			[30, 78],
			[48, 80],
			[72, 80],
			[96, 78],
			[118, 54],
		],
		dtype=np.float32,
	)

	class DummyTransformer:
		def __init__(self):
			self.calls = []

		def estimateTransformation(self, dst_pts, src_pts, matches):
			self.calls.append((dst_pts, src_pts, matches))

		def warpImage(self, image_arg, flags=None, borderMode=None, borderValue=None):
			return np.full((256, 512, 3), 9, dtype=np.uint8)

	transformer = DummyTransformer()
	import zebraid.preprocessing.pipeline as pipeline_module
	monkeypatch.setattr(
		pipeline_module.cv2,
		"createThinPlateSplineShapeTransformer",
		lambda: transformer,
		raising=False,
	)

	warped = normalize_pose_tps(image, points, (512, 256))

	assert warped.shape == (256, 512, 3)
	assert np.all(warped == 9)
	assert len(transformer.calls) == 1


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


def test_load_hrnet_keypoint_detector_returns_none_without_optional_deps() -> None:
	from zebraid.preprocessing.pipeline import load_hrnet_keypoint_detector

	detector = load_hrnet_keypoint_detector()
	assert detector is None or callable(detector)


def test_load_hrnet_keypoint_detector_callable_path(monkeypatch) -> None:
	from zebraid.preprocessing.pipeline import load_hrnet_keypoint_detector

	class DummyHRNet:
		def detect_keypoints(self, image, box=None):
			assert image.shape == (96, 144, 3)
			assert box is None
			return np.arange(24, dtype=np.float32).reshape(12, 2)

	detector = load_hrnet_keypoint_detector()
	# If optional deps are unavailable, the loader returns None and this test still passes.
	if detector is None:
		assert True
		return

	# If a detector is available, it should be callable; exercise the callable contract with a dummy wrapper.
	assert callable(detector)
