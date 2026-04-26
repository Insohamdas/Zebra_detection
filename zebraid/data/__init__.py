"""Data loading and dataset utilities for ZEBRAID."""

from .acquisition import (
	build_path_resolver,
	discover_image_files,
	load_manifest,
	load_records_from_json,
	load_records_from_jsonl,
	load_records_from_csv,
	save_manifest,
)
from .download import download_images
from .dataset_utils import split_dataset, validate_dataset
from .loader import LoadedSample, ZebraDataLoader
from .stream import (
	CCTVStreamConfig,
	LiveFrameRecord,
	StreamError,
	StreamOpenError,
	StreamReadError,
	VideoCaptureStreamSource,
)
from .torch_dataset import ZebraDataset
from .quality import (
	QualityDecision,
	QualityFilterConfig,
	QualityMetrics,
	assess_quality,
	evaluate_quality,
)
from .schema import DataSchemaError, ZebraDataRecord

__all__ = [
	"DataSchemaError",
	"LoadedSample",
	"CCTVStreamConfig",
	"QualityDecision",
	"QualityFilterConfig",
	"QualityMetrics",
	"LiveFrameRecord",
	"ZebraDataLoader",
	"ZebraDataRecord",
	"StreamError",
	"StreamOpenError",
	"StreamReadError",
	"ZebraDataset",
	"assess_quality",
	"build_path_resolver",
	"discover_image_files",
	"download_images",
	"evaluate_quality",
	"load_manifest",
	"load_records_from_json",
	"load_records_from_jsonl",
	"load_records_from_csv",
	"save_manifest",
	"split_dataset",
	"validate_dataset",
	"VideoCaptureStreamSource",
]
