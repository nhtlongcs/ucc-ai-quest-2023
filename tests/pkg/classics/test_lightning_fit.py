# python tests/trainer.py

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from core.opt import CfgNode as CN
from core.opt import Opts
from lightning.pytorch.callbacks import ModelCheckpoint
from pkg.models import MODEL_REGISTRY


def train(model_name, cfg, resume_ckpt=None):
    model = MODEL_REGISTRY.get(model_name)(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./.cache/test/{model_name}",
        verbose=True,
        save_last=True,
    )
    trainer = pl.Trainer(
        default_root_dir="./.cache",
        log_every_n_steps=1,
        max_steps=10,
        max_epochs=2,
        # gpus=None, # debug only
        # strategy=None, # debug only
        devices=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        check_val_every_n_epoch=1,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=16,
        fast_dev_run=False if resume_ckpt is None else True,
        callbacks=[checkpoint_callback] if resume_ckpt is None else [],
        enable_checkpointing=True if resume_ckpt is None else False,
    )
    trainer.fit(model, ckpt_path=resume_ckpt)
    trainer.validate(model, ckpt_path=resume_ckpt)
    del trainer
    del cfg
    del model
    del checkpoint_callback


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "model_name",
    [
        "DeeplabV3ResNet50",
        "SegResNet50",
        "ResNet50VicregL",
        "ResNet50VicregL2",
        "SegResNet101",
        "EffUnet",
        "NestedUnet",
        "ResUnet",
        "Unet",
    ],
)
def test_trainer(model_name):
    bs = 2
    cfg = CN(
        {
            "data": {
                "name": "SegDataset",
                "args": {
                    "SIZE": 380,  # REQUIRED
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
                            "batch_size": 1,
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
    train(model_name, cfg)
    ckpt_path = f"./.cache/test/{model_name}/last.ckpt"
    train(
        model_name,
        cfg,
        resume_ckpt=ckpt_path,
    )


if __name__ == "__main__":
    test_trainer()
