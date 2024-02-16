import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.extractors import EXTRCT_REGISTRY
from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel
from core.utils.device import detach
from core.utils.smooth_losses import MultiLoss
from pkg.models.supervise.convnext_v2_sfnet.encoder import ConvNextEncoder

import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.heads import SFHead


class Model(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.encoder = ConvNextEncoder()
        self.head = SFHead(self.encoder._out_channels, num_classes=num_classes)

    def forward(self, x):
        outs = self.encoder(x)
        out = self.head(outs)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out

@MODEL_REGISTRY.register()
class ConvNext_SFNet(SuperviseModel):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        self.model = Model(num_classes=self.cfg.model["args"]["NUM_CLASS"])
        try:
            # In inference mode, we don't need to compute the loss
            self.loss = MultiLoss(weights=dict(self.cfg.loss["weights"]),
                                smooth_factor=self.cfg.loss["smooth_factor"],
                                convert_to_onehot=True,
                                )
        except:
            pass
    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        logits = self.model(batch["images"])
        # logits = self.normalize_head(preds)
        probs = torch.softmax(logits, dim=1)
        preds_msk = torch.argmax(probs, dim=1)
        return {"logits": logits, "msk": preds_msk}

    def compute_loss(self, logits, batch, return_dict=False, **kwargs):
        targets = batch["labels"].long()
        return self.loss(logits, targets, return_dict=return_dict)

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.cfg.model["args"]["freeze_epochs"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = True

        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss, loss_dict = self.compute_loss(**output, batch=batch, return_dict=True)
        loss = loss.mean()
        # 3. Update monitor
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=False, sync_dist=True, logger=True)

        return {"loss": loss, "log": {"train_loss": loss, **loss_dict}}

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        # add meta data for inference stage
        preds.update({"paths": batch["paths"]})
        return preds
