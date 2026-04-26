import pytest

from zebraid.data import split_dataset, validate_dataset


def test_validate_dataset_accepts_required_fields() -> None:
	dataset = [
		{"image_id": "IMG_001", "gps": "-2.0,34.0"},
		{"image_id": "IMG_002", "gps": "-2.1,34.1"},
	]

	validate_dataset(dataset)


def test_validate_dataset_rejects_missing_fields() -> None:
	with pytest.raises(AssertionError):
		validate_dataset([])

	with pytest.raises(AssertionError):
		validate_dataset([{"image_id": "IMG_001"}])


def test_split_dataset_80_10_10() -> None:
	data = list(range(10))
	train, val, test = split_dataset(data)

	assert train == list(range(8))
	assert val == [8]
	assert test == [9]


def test_split_dataset_validates_ratios() -> None:
	with pytest.raises(ValueError):
		split_dataset([1, 2, 3], train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)