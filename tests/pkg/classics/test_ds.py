from pathlib import Path

import pytest
import torchvision
from core.augmentations import TRANSFORM_REGISTRY
from pkg.datasets import DATASET_REGISTRY
from torch.utils.data import ConcatDataset, DataLoader


@pytest.mark.order(1)
def test_load_dataset(dataset_name="SegDataset", img_size=380):
    image_transform = TRANSFORM_REGISTRY.get("train_classify_tf")(img_size=img_size)
    ds = DATASET_REGISTRY.get(dataset_name)(transform=image_transform)

    item = ds[10]

    # item['image'].shape C,W,H
    # item['mask'].shape W,H

    w, h = item["image"].shape[1], item["image"].shape[2]
    assert (w, h) == (img_size, img_size)
    assert (item["image"].shape[1], item["image"].shape[2]) == (
        item["label"].shape[0],
        item["label"].shape[1],
    )


@pytest.mark.order(2)
def test_batch_dataset(dataset_name="SegDataset", img_size=380, bs=10):
    image_transform = TRANSFORM_REGISTRY.get("train_classify_tf")(img_size=img_size)
    ds = DATASET_REGISTRY.get(dataset_name)(transform=image_transform)
    dl = DataLoader(
        batch_size=bs,
        dataset=ds,
        collate_fn=ds.collate_fn,
        pin_memory=True,  # set to True to enable faster data transfer to CUDA-enabled GPUs.
        # Should be set in conjunction with num_workers > 0. This option take more memory.
    )
    for batch in dl:
        # batch['image'].shape B,C,W,H
        # batch['mask'].shape B,W,H
        w, h = batch["images"].shape[2], batch["images"].shape[3]
        assert (w, h) == (img_size, img_size)
        assert (batch["images"].shape[2], batch["images"].shape[3]) == (
            batch["labels"].shape[1],
            batch["labels"].shape[2],
        )


# @pytest.mark.order(1)
# def test_concat_dataset(dataset_name="ImageFolderFromCSV"):
#     image_transform = TRANSFORM_REGISTRY.get('train_classify_tf')(img_size=380)

#     uds = DATASET_REGISTRY.get(dataset_name)(
#         CSV_PATH="data/train/labels_keyframes_test.csv",
#         IMG_DIR="data/train/keyframes",
#         transform=image_transform,
#         return_lbl=False)
#     ds = DATASET_REGISTRY.get(dataset_name)(
#         CSV_PATH="data/train/labels_keyframes_test.csv",
#         IMG_DIR="data/train/keyframes",
#         transform=image_transform,
#         return_lbl=True)
#     cat_dataset = ConcatDataset([ds, uds])

#     for i in range(len(ds) - 5, len(ds) + 5):
#         print(cat_dataset[i].keys())


# if __name__ == "__main__":
#     test_load_dataset()
