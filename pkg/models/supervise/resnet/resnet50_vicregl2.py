import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from core.extractors import EXTRCT_REGISTRY
from core.models import MODEL_REGISTRY
from core.models.base import SuperviseModel
from core.models.utils import ClassifyBlock
from core.utils.losses import DiceLoss, FocalLoss
from pkg.callbacks import CALLBACKS_REGISTRY
from pkg.datasets import DATASET_REGISTRY
from pkg.metrics import METRIC_REGISTRY


@MODEL_REGISTRY.register()
class ResNet50VicregL2(SuperviseModel):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        # self.net = torchvision.models.segmentation.fcn_resnet101(num_classes=self.cfg.model["args"]["NUM_CLASS"])
        self.net = torchvision.models.segmentation.fcn_resnet50(
            num_classes=self.cfg.model["args"]["NUM_CLASS"]
        )
        ckpt = torch.hub.load("facebookresearch/vicregl:main", "resnet50_alpha0p75")
        self.net.backbone.load_state_dict(ckpt.state_dict())
        # self.loss = DiceLoss(num_classes=self.cfg.model["args"]["NUM_CLASS"])
        self.loss = nn.CrossEntropyLoss()

    def normalize_head(
        self,
        embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        logits = self.net(batch["images"])["out"]
        # logits = self.normalize_head(preds)
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
