from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Source: https://github.com/qubvel/segmentation_models.pytorch
https://huggingface.co/docs/transformers/main/model_doc/mask2former
"""
from semseg.models.heads import SFHead
from transformers import Dinov2Config, Dinov2Model

from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel

available_models = [
    "facebook/dinov2-base",
]


from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor, nn
from torch.hub import load
from torch.nn import functional as F

dino_backbones = {
    "dinov2_s": {"name": "dinov2_vits14", "embedding_size": 384, "patch_size": 14},
    "dinov2_b": {"name": "dinov2_vitb14", "embedding_size": 768, "patch_size": 14},
    "dinov2_l": {"name": "dinov2_vitl14", "embedding_size": 1024, "patch_size": 14},
    "dinov2_g": {"name": "dinov2_vitg14", "embedding_size": 1536, "patch_size": 14},
}


class conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)
        return x


class Segmentor(nn.Module):
    def __init__(
        self, num_classes, backbone="dinov2_s", head="conv", backbones=dino_backbones
    ):
        super(Segmentor, self).__init__()
        self.heads = {"conv": conv_head}
        self.backbones = backbones
        self.backbone = load(
            "facebookresearch/dinov2", self.backbones[backbone]["name"]
        )
        self.backbone.eval()
        self.num_classes = num_classes  # add a class for background if needed
        self.embedding_size = self.backbones[backbone]["embedding_size"]
        self.patch_size = self.backbones[backbone]["patch_size"]
        self.head = self.heads[head](self.embedding_size, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        x = self.backbone.forward_features(x)
        x = x["x_norm_patchtokens"]
        x = x.permute(0, 2, 1)
        x = x.reshape(
            batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1])
        )
        x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class DinoV2(SuperviseModel):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        self.num_classes = self.cfg.model["args"]["NUM_CLASS"]
        self.pretrained = self.cfg.model["args"].get("PRETRAINED", "dinov2_s")
        self.model = Segmentor(self.num_classes, backbone=self.pretrained, head="conv")
        self.loss = nn.CrossEntropyLoss()

    def normalize_head(
        self,
        embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        image_size = batch["images"].shape[-2:]
        logits = self.model(batch["images"])
        logits = F.interpolate(logits, image_size, mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        preds_msk = torch.argmax(probs, dim=1)
        return {"logits": logits, "msk": preds_msk}

    def compute_loss(self, logits, batch, **kwargs):
        return self.loss(logits, batch["labels"].long())

    def predict_step(self, batch, batch_idx=0):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        # add meta data for inference stage
        preds.update({"paths": batch["paths"]})
        return preds
