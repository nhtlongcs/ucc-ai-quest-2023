import abc
from typing import Any

import lightning.pytorch as pl
import torch
import torchvision
from core.dataset import DATASET_REGISTRY
from core.metrics import METRIC_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
from core.utils.device import detach
from torch.utils.data import DataLoader
import rich
from . import MODEL_REGISTRY
from torch.utils.data import ConcatDataset

import lightning as L

from ..utils.schedulers import WarmupPolyLR, RepeatedWarmupPolyLR


class LightningDataModuleWrapper(L.LightningDataModule):
    def __init__(
            self,
            trainloader: torch.utils.data.DataLoader,
            valloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader = None,
    ):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader


@MODEL_REGISTRY.register()
class SuperviseModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.init_model()
        self.learning_rate = self.cfg.get("trainer", {}).get("learning_rate", None)

        if self.learning_rate is None:
            print('[WARNING] learning rate is not defined, auto set to 1e-3')
            self.learning_rate = 1e-3
        self.learning_rate = float(self.learning_rate)

    @abc.abstractmethod
    def init_model(self):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage != "predict":
            image_size = self.cfg["data"]["args"]["SIZE"]
            image_transform_train = TRANSFORM_REGISTRY.get(
                'train_classify_tf')(img_size=image_size)
            image_transform_valid = TRANSFORM_REGISTRY.get('valid_classify_tf')(img_size=image_size)
            # image_transform_test = TRANSFORM_REGISTRY.get('test_classify_tf')(img_size=image_size)

            self.train_dataset = ConcatDataset([
                DATASET_REGISTRY.get(
                    self.cfg["data"]["name"])(
                    **self.cfg["data"]["args"]["train"],
                    data_cfg=self.cfg["data"]["args"],
                    transform=image_transform_train,
                ),
            ])

            self.val_dataset = DATASET_REGISTRY.get(self.cfg["data"]["name"])(
                **self.cfg["data"]["args"]["val"],
                data_cfg=self.cfg["data"]["args"],
                transform=image_transform_valid,
            )

            self.train_dataset.collate_fn = self.val_dataset.collate_fn

            self.metric = [
                METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
                if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
                for mcfg in self.cfg["metric"]
            ]

            self.train_dataloader()
            self.val_dataloader()
            self.init_datamodule()

    def init_datamodule(self):
        self.datamodule = LightningDataModuleWrapper(
            trainloader=self._train_dataloader,
            valloader=self._val_dataloader,
        )

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, visual_embeddings, nlang_embeddings, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch).mean()
        # 3. Update monitor
        self.log("train_loss", detach(loss), prog_bar=True, sync_dist=True)

        return {"loss": loss, "log": {"train_loss": detach(loss)}}

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch)
        # 3. Update metric for each batch
        for m in self.metric:
            m.update(output, batch)

        self.log(
            "val_loss",
            detach(loss),
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": detach(loss)}

    def on_validation_epoch_end(self):

        # 1. Calculate average validation loss
        # loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        # # 2. Calculate metric value
        # out = {"val_loss": loss}
        out = {}
        for m in self.metric:
            # 3. Update metric for each batch
            metric_dict = m.value()
            out.update(metric_dict)
            for k in metric_dict.keys():
                self.log(f"val_{k}", out[k], sync_dist=True)
        # Log string
        # log_string = ""
        # for metric, score in out.items():
        #     if isinstance(score, (int, float)):
        #         log_string += metric + ": " + f"{score:.5f}" + " | "
        # log_string += "\n"
        # print(log_string)
        rich.print(out)

        # 4. Reset metric
        for m in self.metric:
            m.reset()

        # self.log("val_loss", loss.cpu().numpy().item(), sync_dist=True)
        return {**out, "log": out}

    def train_dataloader(self):
        self._train_dataloader = DataLoader(
            **self.cfg["data"]["args"]["train"]["loader"],
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=
            True,  # set to True to enable faster data transfer to CUDA-enabled GPUs. 
            # Should be set in conjunction with num_workers > 0. This option take more memory.
        )
        return self._train_dataloader

    def val_dataloader(self):
        self._val_dataloader = DataLoader(
            **self.cfg["data"]["args"]["val"]["loader"],
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=
            True,  # set to True to enable faster data transfer to CUDA-enabled GPUs. 
            # Should be set in conjunction with num_workers > 0. This option take more memory.
        )
        return self._val_dataloader

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        num_epochs = self.cfg["trainer"]["num_epochs"]
        weight_decay = float(self.cfg["trainer"]["optimizer"]["args"]["weight_decay"])

        if self.cfg["trainer"]["optimizer"]["name"] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=weight_decay)

        else:
            # default optimizer: SGD
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=weight_decay,
                                        nesterov=True)

        if self.cfg["trainer"]["lr_scheduler"]["name"] == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=self.learning_rate,
                                                            epochs=num_epochs,
                                                            steps_per_epoch=len(self.train_dataloader()))
            interval = "step"
        elif self.cfg["trainer"]["lr_scheduler"]["name"] == "Linear":

            lf = lambda x: max(1 - x / num_epochs, 0) * (1.0 - self.learning_rate) + self.learning_rate
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
            interval = "epoch"

        elif self.cfg["trainer"]["lr_scheduler"]["name"] == "WarmupPolyLR":
            scheduler = WarmupPolyLR(optimizer, power=0.9, max_iter=num_epochs, warmup_iter=10, warmup_ratio=0.1,
                                     warmup='linear')
            interval = "epoch"

        elif self.cfg["trainer"]["lr_scheduler"]["name"] == "RepeatedWarmupPolyLR":
            scheduler = RepeatedWarmupPolyLR(optimizer, power=0.9, max_iter=num_epochs, warmup_iter=10,
                                             warmup_ratio=0.1, warmup='linear', restart_warmup_epochs=[10])
            interval = "epoch"

        else:
            # default scheduler: CosineAnnealingWarmRestarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6,
                                                                             last_epoch=-1)
            interval = "step"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            },
        }
