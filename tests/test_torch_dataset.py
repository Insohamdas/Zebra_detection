import numpy as np

from zebraid.data import ZebraDataset


def test_zebra_dataset_returns_image_and_label_without_torch() -> None:
	image = np.zeros((16, 16, 3), dtype=np.uint8)
	dataset = ZebraDataset([(image, 1)])

	item_image, item_label = dataset[0]
	assert item_label == 1
	assert item_image is not None


def test_zebra_dataset_mapping_input() -> None:
	image = np.zeros((8, 8, 3), dtype=np.uint8)
	dataset = ZebraDataset([{"image": image, "label": 7}])

	item_image, item_label = dataset[0]
	assert item_label == 7
	assert item_image is not None