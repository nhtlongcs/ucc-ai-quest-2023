import torch
from core.opt import Opts

from pathlib import Path
from core.augmentations import TRANSFORM_REGISTRY
from pkg.models import MODEL_REGISTRY
from pkg.datasets import DATASET_REGISTRY
from tqdm import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import lightning.pytorch as pl

def check(cfg, pretrained_ckpt=None):
    model = MODEL_REGISTRY.get(cfg.model["name"]).load_from_checkpoint(pretrained_ckpt,
                                                                        config=cfg,
                                                                        strict=True)
    trainer = pl.Trainer(
        devices=1, num_nodes=1,
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
    )
    trainer.validate(model)
    del trainer
    del cfg
    del model

def main():
    cfg = Opts().parse_args()
    check(cfg, pretrained_ckpt=cfg['global']['pretrained'])

if __name__ == "__main__":
    main()