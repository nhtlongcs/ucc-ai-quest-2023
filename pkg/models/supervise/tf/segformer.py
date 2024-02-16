from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel
from core.utils.smooth_losses import MultiLoss


@MODEL_REGISTRY.register()
class Segformer(SuperviseModel):
    def __init__(self, config):
        self.pretrained = "nvidia/segformer-b3-finetuned-ade-512-512"
        super().__init__(config)
        if self.cfg.model["args"].get("PRETRAINED", None) is not None:
            self.pretrained = self.cfg.model["args"]["PRETRAINED"]
            self.init_model()

    def init_model(self):
        self.num_classes = self.cfg.model["args"]["NUM_CLASS"]
        self._cfg = SegformerConfig().from_pretrained(self.pretrained)
        self._cfg.update(
            {"output_hidden_states": False, "num_labels": self.num_classes}
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.pretrained,
            config=self._cfg,
            ignore_mismatched_sizes=True,
        )

        try:
            # In inference mode, we don't need to compute the loss
            self.loss = MultiLoss(weights=dict(self.cfg.loss["weights"]),
                                smooth_factor=self.cfg.loss["smooth_factor"],
                                convert_to_onehot=True,
                                )
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
        logits = self.model(pixel_values=batch["images"]).logits
        # shape (batch_size, num_labels, height/4, width/4)
        # logits = self.normalize_head(preds)
        logits = F.interpolate(logits, image_size, mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        preds_msk = torch.argmax(probs, dim=1)
        return {"logits": logits, "msk": preds_msk}

    def compute_loss(self, logits, batch, return_dict=False, **kwargs):
        targets = batch["labels"].long()
        return self.loss(logits, targets, return_dict=return_dict)

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        # add meta data for inference stage
        preds.update({"paths": batch["paths"]})
        return preds

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.cfg.model["args"]["freeze_epochs"]:
            for param in self.model.segformer.parameters():
                param.requires_grad = False
        else:
            for param in self.model.segformer.parameters():
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
