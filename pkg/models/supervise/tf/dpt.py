from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DPTConfig, DPTForSemanticSegmentation

from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel


@MODEL_REGISTRY.register()
class DPT(SuperviseModel):
    def __init__(self, config):
        self.pretrained = "Intel/dpt-large-ade"
        super().__init__(config)
        if self.cfg.model["args"].get("PRETRAINED", None) is not None:
            self.pretrained = self.cfg.model["args"]["PRETRAINED"]
            self.init_model()

    def init_model(self):
        self.num_classes = self.cfg.model["args"]["NUM_CLASS"]
        self._cfg = DPTConfig().from_pretrained(self.pretrained)
        self._cfg.update(
            {"output_hidden_states": False, "num_labels": self.num_classes}
        )
        self.model = DPTForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.pretrained,
            config=self._cfg,
            ignore_mismatched_sizes=True,
        )
        self.loss = nn.CrossEntropyLoss()

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

    def compute_loss(self, logits, batch, **kwargs):
        return self.loss(logits, batch["labels"].long())

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        assert (
            batch["images"].shape[0] == 1
        ), "Batch size must be 1 for this model, sorry! ugly code"
        preds = self.forward(batch)
        # add meta data for inference stage
        sz = batch["sizes"][0]
        msk = preds["msk"][0].unsqueeze(0).unsqueeze(0).float()
        origin_msk = F.interpolate(msk, sz, mode="bilinear", align_corners=False).long()
        preds["msk"] = origin_msk.squeeze(0)
        preds.update({"paths": batch["paths"]})
        return preds
