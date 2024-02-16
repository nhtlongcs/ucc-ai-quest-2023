import os
from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from core.dataset import DATASET_REGISTRY
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@DATASET_REGISTRY.register()
class SegDataset(Dataset):
    def __init__(
            self,
            root_dir="data",
            phase="warmup",
            split="train",
            transform=None,  # MUST HAVE
            **kwargs,  # MUST HAVE
    ):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / phase / "img" / split
        self.ann_dir = self.root_dir / phase / "ann" / split
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)
        self.classnames = ["background", "plants"]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_dir / self.img_list[idx]))
        ann_path = self.ann_dir / f"{Path(self.img_list[idx]).stem}.png"
        ann = np.array(Image.open(ann_path))
        w, h = ann.shape[0], ann[1]

        item = {
            "label": ann,
            "image": img,
            # 'origin_h': h, # not use
            # 'origin_w': w, # not use
            "path": self.img_dir / self.img_list[idx],
        }
        if self.transform:
            augmented = self.transform(image=img, mask=ann)
            item["image"] = augmented["image"]
            item["label"] = augmented["mask"]

        assert (item["image"].shape[1], item["image"].shape[2]) == (
            item["label"].shape[0],
            item["label"].shape[1],
        ), "image label should be have the same size"
        return item

    def collate_fn(self, batch):
        batch_dict = {
            "images": torch.stack([x["image"] for x in batch]),
            "labels": torch.stack([x["label"] for x in batch]),
            # "origin_w": [x['origin_w'] for x in batch],
            # "origin_h": [x['origin_h'] for x in batch],
            "paths": [x["path"] for x in batch],
        }
        return batch_dict


@DATASET_REGISTRY.register()
class TestImageDataset(Dataset):
    def __init__(self, IMG_DIR, transform=None, **kwargs):  # MUST HAVE  # MUST HAVE
        self.img_dir = Path(IMG_DIR)
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_dir / self.img_list[idx]))

        item = {
            "image": img,
            "path": self.img_dir / self.img_list[idx],
            "size": img.shape[:2],
        }
        if self.transform:
            augmented = self.transform(image=img)
            item["image"] = augmented["image"]

        return item

    def collate_fn(self, batch):
        batch_dict = {
            "images": torch.stack([x["image"] for x in batch]),
            "paths": [x["path"] for x in batch],
            "sizes": [x["size"] for x in batch],
        }
        return batch_dict


class SegDataModule(pl.LightningDataModule):
    def __init__(self, root_dir="data", phase="warmup", batch_size: int = 8):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="train",
            transform=A.Compose([A.RandomCrop(380, 380), ToTensorV2()]),
        )
        self.valid = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="valid",
            transform=ToTensorV2(),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False)


@DATASET_REGISTRY.register()
class KFoldSegDatasetCSV(Dataset):
    def __init__(self, csv_path: str,
                 fold_index: int = 0,
                 split="train",
                 transform=None,  # MUST HAVE
                 **kwargs  # MUST HAVE
                 ):
        self.csv_path = pd.read_csv(csv_path)
        self.transform = transform
        self.classnames = ['background', 'plants']

        df = pd.read_csv(csv_path)
        if split == "train":
            df = df[df["fold"] != fold_index]
        elif split == "valid":
            df = df[df["fold"] == fold_index]
        else:
            raise ValueError(f"split must be train or valid, but got {split}")
        self.df = df
        self.sanity_check()

    def sanity_check(self):
        # check if the csv is correct
        assert self.df["img"].apply(lambda x: Path(x).exists()).all(), "Some images references in the CSV do not exist"
        assert self.df["ann"].apply(
            lambda x: Path(x).exists()).all(), "Some annotations references in the CSV do not exist"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # img = np.array(Image.open(self.img_dir / self.img_list[idx]))
        img = np.array(Image.open(self.df.iloc[idx]["img"]))
        # ann_path = self.ann_dir / f"{Path(self.img_list[idx]).stem}.png
        # ann = np.array(Image.open(ann_path))
        ann = np.array(Image.open(self.df.iloc[idx]["ann"]))
        w, h = ann.shape[0], ann[1]

        item = {
            'label': ann,
            'image': img,
            # 'origin_h': h, # not use
            # 'origin_w': w, # not use
            # 'path': self.img_dir / self.img_list[idx],
            'path': self.df.iloc[idx]["img"],
        }
        if self.transform:
            augmented = self.transform(image=img, mask=ann)
            item['image'] = augmented["image"]
            item['label'] = augmented["mask"]

        assert (item['image'].shape[1], item['image'].shape[2]) == (
            item['label'].shape[0], item['label'].shape[1]), "image label should be have the same size"
        return item

    def collate_fn(self, batch):
        batch_dict = {
            "images": torch.stack([x['image'] for x in batch]),
            "labels": torch.stack([x['label'] for x in batch]),
            # "origin_w": [x['origin_w'] for x in batch],
            # "origin_h": [x['origin_h'] for x in batch],
            "paths": [x['path'] for x in batch],
        }
        return batch_dict


if __name__ == "__main__":
    ds = SegDataset()
    item = ds[10]
    print("Done!")
