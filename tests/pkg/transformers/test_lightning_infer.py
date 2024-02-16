# python tests/trainer.py

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
import torchvision
from core.augmentations import TRANSFORM_REGISTRY
from core.dataset import DATASET_REGISTRY
from core.opt import CfgNode as CN
from core.opt import Opts
from lightning.pytorch.callbacks import ModelCheckpoint
from pkg.models import MODEL_REGISTRY
from torch.utils.data import DataLoader, Dataset


def predict(model_name, cfg, resume_ckpt=None, IMG_SIZE=380):
    class WrapperDataModule(pl.LightningDataModule):
        def __init__(self, ds, batch_size):
            super().__init__()
            self.ds = ds
            self.batch_size = batch_size

        def predict_dataloader(self):
            return DataLoader(
                self.ds,
                batch_size=self.batch_size,
                collate_fn=self.ds.collate_fn,
            )

    model = MODEL_REGISTRY.get(model_name).load_from_checkpoint(
        resume_ckpt, config=cfg, strict=True
    )

    trainer = pl.Trainer(
        devices=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        enable_checkpointing=False,
    )
    image_transform = TRANSFORM_REGISTRY.get("test_classify_tf")(img_size=IMG_SIZE)

    ds = DATASET_REGISTRY.get(cfg["data"]["name"])(
        **cfg.data["args"]["val"],
        transform=image_transform,
    )
    dm = WrapperDataModule(ds, batch_size=1)
    prds = trainer.predict(model, dm)


@pytest.mark.order(5)
@pytest.mark.parametrize(
    "model_name, IMG_SIZE",
    [
        ("Segformer", 380),
        ("Mask2former", 380),
    ],
)
def test_predict(model_name, IMG_SIZE):
    bs = 2
    cfg = CN(
        {
            "data": {
                "name": "SegDataset",
                "args": {
                    "SIZE": IMG_SIZE,  # REQUIRED
                    "train": {
                        "root_dir": "data",
                        "phase": "warmup",
                        "split": "valid",
                        "loader": {
                            "batch_size": bs,
                            "num_workers": 2,
                        },
                    },
                    "val": {
                        "root_dir": "data",
                        "phase": "warmup",
                        "split": "valid",
                        "loader": {
                            "batch_size": bs,
                            "num_workers": 2,
                            "shuffle": False,
                            "drop_last": False,
                        },
                    },
                },
            },
            "model": {
                "name": model_name,
                "args": {
                    "NUM_CLASS": 2,
                },
            },
            "metric": [
                {"name": "SMAPIoUMetricWrapper", "args": {"label_key": "labels"}}
            ],
        }
    )
    ckpt_path = f"./.cache/test/{model_name}/last.ckpt"

    predict(
        model_name,
        cfg,
        resume_ckpt=ckpt_path,
        IMG_SIZE=IMG_SIZE,
    )


if __name__ == "__main__":
    test_predict()
