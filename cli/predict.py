import json
import os.path as osp
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from core.augmentations import TRANSFORM_REGISTRY
from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel
from core.opt import Opts
from pkg.datasets import DATASET_REGISTRY
from pkg.datasets.ucc import TestImageDataset
from pkg.models import MODEL_REGISTRY
from pkg.utils import mask_to_rgb, mask_to_rle, show_in_grid


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
            num_workers=20,
        )


class Predictor:
    def __init__(
        self,
        model: SuperviseModel,
        cfg: Opts,
        batch_size: int = 1,
    ):
        self.cfg = cfg
        self.model = model
        self.threshold = 0.5
        self.batch_size = batch_size
        self.setup()

    def setup(self):
        image_size = self.cfg["data"]["SIZE"]
        transform = TRANSFORM_REGISTRY.get("test_classify_tf")(img_size=image_size)
        self.ds = TestImageDataset(**self.cfg.data, transform=transform, num_rows=-1)

    def predict(self):
        trainer = pl.Trainer(
            devices=1 if torch.cuda.device_count() else None,  # Use all gpus available
            num_nodes=1,
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
            sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
            enable_checkpointing=False,
        )

        dm = WrapperDataModule(self.ds, batch_size=self.batch_size)
        prds = trainer.predict(self.model, datamodule=dm)
        return prds

    def predict_json(self):
        prds = self.predict()
        results = {}
        for _, batch_prds in enumerate(prds):
            pred_msk = batch_prds["msk"]  # batch ids
            path = batch_prds["paths"]

            for pred, path in list(zip(pred_msk, path)):
                filename = path.name
                rle = mask_to_rle(pred)
                results[filename] = {
                    "counts": rle,
                    "height": pred.shape[0],
                    "width": pred.shape[1],
                }
        return results


def main():
    cfg = Opts().parse_args()
    resume_ckpt = cfg["global"]["pretrained"]
    save_path = Path(cfg["global"].get("save_path", "./"))
    batch_sz = cfg["global"]["batch_size"]
    # if save_path is directory, then savepath = savepath / 'predict.csv'
    if save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / "predict.json"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        assert save_path.suffix == ".json", f"Path {save_path} must be a json file"

    model = MODEL_REGISTRY.get(cfg.model["name"]).load_from_checkpoint(
        resume_ckpt, config=cfg, strict=True
    )
    p = Predictor(model, cfg, batch_size=batch_sz)
    results = p.predict_json()
    with open(save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
