from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.smooth_losses import MultiLoss

"""
Source: https://github.com/qubvel/segmentation_models.pytorch
https://huggingface.co/docs/transformers/main/model_doc/mask2former
"""
from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)

from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel

available_models = [
    "facebook/mask2former-swin-base-ade-semantic",
    "facebook/mask2former-swin-base-IN21k-ade-semantic",
    "facebook/mask2former-swin-large-ade-semantic",
    "facebook/mask2former-swin-small-ade-semantic",
    "facebook/mask2former-swin-tiny-ade-semantic",
]


@MODEL_REGISTRY.register()
class Mask2former(SuperviseModel):
    def __init__(self, config):
        # self.pretrained = "facebook/mask2former-swin-base-IN21k-ade-semantic"
        self.pretrained = "facebook/mask2former-swin-base-ade-semantic"
        super().__init__(config)
        if self.cfg.model["args"].get("PRETRAINED", None) is not None:
            self.pretrained = self.cfg.model["args"]["PRETRAINED"]
            self.init_model()

    def init_model(self):
        self.num_classes = self.cfg.model["args"]["NUM_CLASS"]
        self._cfg = Mask2FormerConfig().from_pretrained(self.pretrained)
        self._cfg.update({"output_hidden_states": False, "num_queries": 2})
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.pretrained,
            config=self._cfg,
            ignore_mismatched_sizes=True,
        )

        try:
            # In inference mode, we don't need to compute the loss
            self.loss = nn.CrossEntropyLoss(label_smoothing=self.cfg.loss["smooth_factor"])
        except:
            pass 

    def normalize_head(
        self,
        embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        image_size = batch["images"].shape[-2:]
        logits = self.model(pixel_values=batch["images"]).masks_queries_logits
        # shape (batch_size, num_labels, height/4, width/4)
        # logits = self.normalize_head(preds)
        logits = F.interpolate(logits, image_size, mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)

        preds_msk = torch.argmax(probs, dim=1)
        return {"logits": logits, "msk": preds_msk}

    def compute_loss(self, logits, batch, **kwargs):
        targets = batch["labels"].long()
        return self.loss(logits, targets)

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        # add meta data for inference stage
        preds.update({"paths": batch["paths"]})
        return preds

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.cfg.model["args"]["freeze_epochs"]:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch)
        loss = loss.mean()
        # 3. Update monitor
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return {"loss": loss, "log": {"train_loss": loss}}